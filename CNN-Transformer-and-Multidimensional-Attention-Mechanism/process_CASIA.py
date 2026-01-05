#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CASIA 数据处理脚本（供 train_CASIA.py 调用）

目标：
1) MFCC 帧数对齐：让 1.8s segment 的 MFCC time 维稳定在 ~57（默认 hop_length=512 对齐 librosa 默认 hop）
2) 支持两种划分：
   - speaker-independent：test_speakers 指定哪些 speaker 作为测试（留人测试）
   - random split：test_speakers 为空/None 时，按 split_rate 做随机划分（更接近论文里的 80/20 随机划分）
"""

import os
import glob
import math
import json
import random
import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import soundfile as sf
from scipy.signal import resample_poly
from python_speech_features import mfcc as psf_mfcc
from python_speech_features import logfbank, fbank, sigproc


# =========================
# CASIA label mapping
# 文件夹名：angry fear happy neutral sad surprise
# =========================
CASIA_LABEL: Dict[str, int] = {
    "neutral": 0,
    "sad": 1,
    "angry": 2,
    "happy": 3,
    "fear": 4,
    "surprise": 5,
}


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """读取 wav 并重采样到 target_sr（避免 librosa/numba 链路）"""
    x, sr = sf.read(path, always_2d=False)
    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)

    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        up = target_sr // g
        down = sr // g
        x = resample_poly(x, up, down).astype(np.float32)
    return x


def _pad_or_trim_time(feat_ct: np.ndarray, target_frames: int) -> np.ndarray:
    """
    feat_ct: [C, T]
    target_frames: 目标 T
    返回: [C, target_frames]
    """
    if target_frames is None:
        return feat_ct
    c, t = feat_ct.shape
    if t == target_frames:
        return feat_ct
    if t > target_frames:
        # center crop
        start = (t - target_frames) // 2
        return feat_ct[:, start:start + target_frames]
    # pad at end with zeros
    pad = target_frames - t
    return np.pad(feat_ct, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)


class FeatureExtractor(object):
    """
    关键：MFCC 的 hop_length 默认 512（对应 librosa 默认 hop_length），
    并在输出后强制对齐到 target_frames（默认按 segment_length/hop_length 自动算）。
    """
    def __init__(
        self,
        sample_rate: int,
        nmfcc: int = 26,
        hop_length: int = 512,           # samples
        target_frames: Optional[int] = None,
        segment_length: float = 1.8,
    ):
        self.sample_rate = sample_rate
        self.nmfcc = nmfcc
        self.hop_length = hop_length
        self.segment_length = segment_length

        if target_frames is None:
            # 1.8s * 16000 / 512 = 56.25 -> ceil = 57（与原工程注释一致）
            target_frames = int(math.ceil(segment_length * sample_rate / hop_length))
        self.target_frames = target_frames

    def get_features(self, features_to_use: str, X: np.ndarray) -> np.ndarray:
        accepted = ("logfbank", "mfcc", "fbank", "melspectrogram", "spectrogram", "interspeech2018")
        if features_to_use not in accepted:
            raise NotImplementedError(f"{features_to_use} not in {accepted}!")

        if features_to_use == "logfbank":
            return self.get_logfbank(X)
        if features_to_use == "mfcc":
            return self.get_mfcc(X, self.nmfcc)
        if features_to_use == "fbank":
            return self.get_fbank(X)
        if features_to_use == "melspectrogram":
            return self.get_melspectrogram(X)
        if features_to_use == "spectrogram":
            return self.get_spectrogram(X)
        if features_to_use == "interspeech2018":
            return self.get_spectrogram_interspeech2018(X)
        raise RuntimeError("Unreachable")

    def get_logfbank(self, X: np.ndarray) -> np.ndarray:
        def _get_logfbank(x):
            out = logfbank(
                signal=x,
                samplerate=self.sample_rate,
                winlen=0.040,
                winstep=self.hop_length / self.sample_rate,  # 与 mfcc 帧率一致
                nfft=1024,
                highfreq=4000,
                nfilt=40,
            )  # [T, 40]
            out = out.T  # -> [40, T]
            return _pad_or_trim_time(out, self.target_frames)

        return np.apply_along_axis(_get_logfbank, 1, X)

    def get_mfcc(self, X: np.ndarray, nmfcc: int = 26) -> np.ndarray:
        def _get_mfcc(x):
            feat = psf_mfcc(
                signal=x,
                samplerate=self.sample_rate,
                winlen=0.040,
                winstep=self.hop_length / self.sample_rate,  # 512/16000≈0.032
                numcep=nmfcc,
                nfilt=nmfcc,
                nfft=1024,
                preemph=0.97,
                appendEnergy=True,
            )  # [T, nmfcc]
            feat = feat.T  # -> [nmfcc, T]
            return _pad_or_trim_time(feat, self.target_frames)

        return np.apply_along_axis(_get_mfcc, 1, X)

    def get_fbank(self, X: np.ndarray) -> np.ndarray:
        def _get_fbank(x):
            out, _ = fbank(
                signal=x,
                samplerate=self.sample_rate,
                winlen=0.040,
                winstep=self.hop_length / self.sample_rate,
                nfft=1024,
            )  # [T, F]
            out = out.T
            return _pad_or_trim_time(out, self.target_frames)

        return np.apply_along_axis(_get_fbank, 1, X)

    def get_melspectrogram(self, X: np.ndarray) -> np.ndarray:
        """
        不依赖 librosa：用 logfbank 近似 log-Mel-spectrogram
        """
        def _get_mel(x):
            mel = logfbank(
                signal=x,
                samplerate=self.sample_rate,
                winlen=0.040,
                winstep=self.hop_length / self.sample_rate,
                nfft=1024,
                nfilt=80,
                highfreq=self.sample_rate / 2,
            )  # [T, 80]
            mel = mel.T  # [80, T]
            return _pad_or_trim_time(mel, self.target_frames)

        return np.apply_along_axis(_get_mel, 1, X)

    def get_spectrogram(self, X: np.ndarray) -> np.ndarray:
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        return np.apply_along_axis(_get_spectrogram, 1, X)

    def get_spectrogram_interspeech2018(self, X: np.ndarray) -> np.ndarray:
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.magspec(frames, NFFT=3198)
            out = out / out.max() * 2 - 1
            out = np.sign(out) * np.log(1 + 255 * np.abs(out)) / np.log(256)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        return np.apply_along_axis(_get_spectrogram, 1, X)


def segment(
    wavfile,
    sample_rate: int = 16000,
    segment_length: float = 1.8,
    overlap: float = 1.6,
    padding=None,
) -> Optional[np.ndarray]:
    """
    返回 shape: [num_segments, seg_len] 的 ndarray
    """
    if isinstance(wavfile, str):
        wav_data = load_wav(wavfile, target_sr=sample_rate)
    elif isinstance(wavfile, np.ndarray):
        wav_data = wavfile
    else:
        raise TypeError(f"Type {type(wavfile)} not supported.")

    seg_wav_len = int(segment_length * sample_rate)
    wav_len = len(wav_data)

    if seg_wav_len > wav_len:
        if padding:
            n = math.ceil(seg_wav_len / wav_len)
            wav_data = np.hstack(n * [wav_data])
            wav_len = len(wav_data)
        else:
            return None

    if (segment_length - overlap) <= 0:
        raise ValueError("segment_length - overlap must be > 0")

    step = int((segment_length - overlap) * sample_rate)

    X = []
    index = 0
    while index + seg_wav_len <= wav_len:
        X.append(wav_data[index: index + seg_wav_len])
        index += step

    if len(X) == 0:
        return None
    return np.array(X)


def _stratified_split(
    records: List[Tuple[np.ndarray, int]],
    split_rate: float,
    seed: int,
) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    """
    为了“更好看且更稳定”，这里做按标签分层的随机划分（仍然是随机 80/20）。
    如果你希望完全不分层，把这段换成纯 np.random.choice 即可。
    """
    rng = np.random.RandomState(seed)
    by_label: Dict[int, List[Tuple[np.ndarray, int]]] = {}
    for wav, y in records:
        by_label.setdefault(y, []).append((wav, y))

    train_files, valid_files = [], []
    for y, items in by_label.items():
        n = len(items)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = int(round(n * split_rate))
        for i in idx[:n_train]:
            train_files.append(items[i])
        for i in idx[n_train:]:
            valid_files.append(items[i])

    rng.shuffle(train_files)
    rng.shuffle(valid_files)
    return train_files, valid_files


def _extract_features_for_split(
    train_files: List[Tuple[np.ndarray, int]],
    valid_files: List[Tuple[np.ndarray, int]],
    *,
    num_label: Optional[Dict[int, int]],
    features_to_use: str,
    sample_rate: int,
    nmfcc: int,
    hop_length: int,
    target_frames: Optional[int],
    train_overlap: float,
    test_overlap: float,
    segment_length: float,
    featuresFileName: str,
    toSaveFeatures: bool,
    aug: Optional[str],
    padding,
):
    get_label = lambda x: x[1]
    info = {
        "train": json.dumps(Counter(map(get_label, train_files))),
        "test": json.dumps(Counter(map(get_label, valid_files))),
    }

    if num_label is not None:
        print(f"Amount of categories: {len(num_label)}")
    print(f"Training Datasets: {len(train_files)}, Testing Datasets: {len(valid_files)}")

    if aug == "upsampling":
        label_wav = {k: [] for k in set([lb for _, lb in train_files])}
        for wav, label in train_files:
            label_wav[label].append(wav)
        maxval = max(len(v) for v in label_wav.values())
        for label, wavs in label_wav.items():
            nw = len(wavs)
            if nw == 0:
                continue
            indices = list(np.random.choice(range(nw), maxval - nw, replace=True))
            for i in indices:
                train_files.append((wavs[i], label))
        random.shuffle(train_files)
        print(f"After Augmentation...\nTraining Datasets: {len(train_files)}, Testing Datasets: {len(valid_files)}")

    feature_extractor = FeatureExtractor(
        sample_rate=sample_rate,
        nmfcc=nmfcc,
        hop_length=hop_length,
        target_frames=target_frames,
        segment_length=segment_length,
    )

    print("Extracting features for training datasets")
    train_X, train_y = [], []
    for wav_data, label in tqdm(train_files):
        X1 = segment(
            wav_data,
            sample_rate=sample_rate,
            segment_length=segment_length,
            overlap=train_overlap,
            padding=padding,
        )
        if X1 is None:
            continue
        y1 = len(X1) * [label]
        X1 = feature_extractor.get_features(features_to_use, X1)  # [nseg, C, T]
        train_X.append(X1)
        train_y += y1

    train_X = np.row_stack(train_X)
    train_y = np.array(train_y)
    assert len(train_X) == len(train_y), f"X/y mismatch: {train_X.shape} vs {train_y.shape}"
    print(f"Amount of categories after segmentation(training): {Counter(train_y).items()}")

    print("Extracting features for test datasets")
    val_dict = []
    test_y = []
    if test_overlap >= segment_length:
        test_overlap = segment_length / 2

    for wav_data, label in tqdm(valid_files):
        X1 = segment(
            wav_data,
            sample_rate=sample_rate,
            segment_length=segment_length,
            overlap=test_overlap,
            padding=padding,
        )
        if X1 is None:
            continue
        X1 = feature_extractor.get_features(features_to_use, X1)
        val_dict.append({"X": X1, "y": label})
        test_y.append(label)

    print(f"Amount of categories after segmentation(test): {Counter(test_y).items()}")
    info["train_seg"] = f"{Counter(train_y).items()}"
    info["mfcc_target_frames"] = str(feature_extractor.target_frames)
    info["mfcc_hop_length"] = str(feature_extractor.hop_length)

    if toSaveFeatures:
        print(f"Saving features to {featuresFileName}.")
        features = {"train_X": train_X, "train_y": train_y, "val_dict": val_dict, "info": info}
        with open(featuresFileName, "wb") as f:
            pickle.dump(features, f)

    return train_X, train_y, val_dict, info


def process_CASIA(
    datasets_root: str,
    LABEL_DICT: Dict[str, int],
    datadir: str = "data/CASIA",
    featuresFileName: Optional[str] = None,
    features_to_use: str = "mfcc",
    impro_or_script: str = "CASIA",
    sample_rate: int = 16000,
    nmfcc: int = 26,
    hop_length: int = 512,
    target_frames: Optional[int] = None,
    train_overlap: float = 1.6,
    test_overlap: float = 1.6,
    segment_length: float = 1.8,
    split_rate: float = 0.8,
    toSaveFeatures: bool = True,
    aug: Optional[str] = None,
    padding=None,
    test_speakers: Optional[List[str]] = None,
    speakers: Optional[List[str]] = None,
    seed: int = 987654,
    **kwargs,
):
    """
    datasets_root: 例如 E:\\SER0103\\bc5e6\\casia

    - speaker split：test_speakers = ["spk4"]（留人测试）
    - random split：test_speakers 为空/None（更接近论文随机划分）
    """
    os.makedirs(datadir, exist_ok=True)

    if featuresFileName is None:
        featuresFileName = f"{datadir}/features_{features_to_use}_{impro_or_script}.pkl"

    if not os.path.exists(datasets_root):
        raise FileNotFoundError(f"{datasets_root} not existed.")

    # speakers list
    if speakers is None:
        speakers = sorted([d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))])

    # collect wavs
    records_spk: List[Tuple[str, np.ndarray, int]] = []
    num_label: Dict[int, int] = {}

    for spk in speakers:
        spk_dir = os.path.join(datasets_root, spk)
        if not os.path.isdir(spk_dir):
            continue

        for emo in sorted(os.listdir(spk_dir)):
            emo_dir = os.path.join(spk_dir, emo)
            if not os.path.isdir(emo_dir):
                continue
            if emo not in LABEL_DICT:
                continue

            wav_list = sorted(glob.glob(os.path.join(emo_dir, "*.wav")))
            for wav_path in wav_list:
                wav_data = load_wav(wav_path, target_sr=sample_rate)

                seg_wav_len = int(segment_length * sample_rate)
                if len(wav_data) < seg_wav_len and not padding:
                    continue

                label_id = LABEL_DICT[emo]
                records_spk.append((spk, wav_data, label_id))
                num_label[label_id] = num_label.get(label_id, 0) + 1

    if len(records_spk) == 0:
        raise RuntimeError("No wav files found. Please check datasets_root / folder names.")

    # split
    if test_speakers:
        test_speakers = [s.strip() for s in test_speakers if s.strip()]
        train_files = [(wav, y) for spk, wav, y in records_spk if spk not in test_speakers]
        valid_files = [(wav, y) for spk, wav, y in records_spk if spk in test_speakers]
        if len(valid_files) == 0:
            raise RuntimeError(
                f"test_speakers={test_speakers} matched nothing. "
                f"Available: {sorted(set([r[0] for r in records_spk]))}"
            )
    else:
        # 论文“随机 80/20”更接近这条路径（speaker-mixed）
        records = [(wav, y) for _, wav, y in records_spk]
        train_files, valid_files = _stratified_split(records, split_rate=split_rate, seed=seed)

    return _extract_features_for_split(
        train_files=train_files,
        valid_files=valid_files,
        num_label=num_label,
        features_to_use=features_to_use,
        sample_rate=sample_rate,
        nmfcc=nmfcc,
        hop_length=hop_length,
        target_frames=target_frames,
        train_overlap=train_overlap,
        test_overlap=test_overlap,
        segment_length=segment_length,
        featuresFileName=featuresFileName,
        toSaveFeatures=toSaveFeatures,
        aug=aug,
        padding=padding,
    )
