#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CASIA 训练入口（参考 train_Emodb.py 的训练策略）：
- 1.8s segment，overlap 1.6s
- MFCC 26 维（process_CASIA.py 内部将 time 维对齐到 ~57）
- mixup alpha=0.2
- Adam, lr=0.001, 指数衰减 0.95，最小 lr=1e-6
- 默认使用论文式的随机 80/20 划分（test_speakers 为空）
- 训练指令：
- python train_CASIA.py --dataset_root E:\SER0103\bc5e6\casia -d data\CASIA 2>&1 | Tee-Object -FilePath data\CASIA\run.log
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import time
import json
import argparse
from collections import Counter

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# 你环境里 timm==0.6.7 可用；若你又换回更老 torch 导致 timm 不能导入，再改成自写的 label smoothing
from timm.loss import LabelSmoothingCrossEntropy

import models
import data_loader
from process_CASIA import CASIA_LABEL, process_CASIA


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Train on CASIA")

    # features
    parser.add_argument("-f", "--features_to_use", default="mfcc", type=str,
                        help='{"mfcc","logfbank","fbank","spectrogram","melspectrogram"}')
    parser.add_argument("-i", "--impro_or_script", default="CASIA", type=str)
    parser.add_argument("-s", "--sample_rate", default=16000, type=int)
    parser.add_argument("-n", "--nmfcc", default=26, type=int)

    # MFCC frame alignment (关键)
    parser.add_argument("--hop_length", default=512, type=int, help="MFCC hop length in samples (default 512)")
    parser.add_argument("--target_frames", default=None, type=int, help="Force MFCC time frames; default auto")

    parser.add_argument("--train_overlap", default=1.6, type=float)
    parser.add_argument("--test_overlap", default=1.6, type=float)
    parser.add_argument("--segment_length", default=1.8, type=float)
    parser.add_argument("--toSaveFeatures", default=True, type=bool)
    parser.add_argument("--loadFeatures", default=True, type=bool)
    parser.add_argument("--featuresFileName", default=None, type=str)

    # dataset root (CASIA)
    parser.add_argument("--dataset_root", default=r"E:\SER0103\bc5e6\casia", type=str)

    # 关键：默认空 => 随机划分（论文式 80/20）
    parser.add_argument("--test_speakers", default="", type=str,
                        help='Speaker split, e.g. "spk4". Empty => random split (paper-like)')

    # model
    parser.add_argument("-m", "--model", default="CTMAM_EMODB", type=str)
    parser.add_argument("--num_classes", default=6, type=int)
    parser.add_argument("--SaveModel", default=True, type=bool)

    # random split rate (only used when test_speakers is empty)
    parser.add_argument("--split_rate", default=0.8, type=float)
    parser.add_argument("--aug", default=None, type=str)
    parser.add_argument("--padding", default=True, type=bool)

    # training
    parser.add_argument("--seed", default=987654, type=int)
    parser.add_argument("-b", "--batch_size", default=128, type=int)
    parser.add_argument("-l", "--learning_rate", default=0.001, type=float)
    parser.add_argument("--lr_min", default=1e-6, type=float)
    parser.add_argument("--lr_schedule", default="exp", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("-e", "--epochs", default=150, type=int)
    parser.add_argument("--iter", default=1, type=int)
    parser.add_argument("-g", "--gpu", default=0, type=int)
    parser.add_argument("--weight", default=False, type=bool)
    parser.add_argument("--alpha", default=0.2, type=float)

    # save dir
    parser.add_argument("-d", "--datadir", default="data/CASIA", type=str)

    return parser.parse_args()


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mixup_data(x, y, alpha=0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda() if torch.cuda.is_available() else torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    mixed_x, y_a, y_b = map(Variable, (mixed_x, y_a, y_b))
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_loss(model, criterion, x, y, alpha):
    if alpha > 0:
        mix_x, ya, yb, lam = mixup_data(x, y, alpha)
        if torch.cuda.is_available():
            mix_x, ya, yb = mix_x.cuda(), ya.cuda(), yb.cuda()
        ya = ya.squeeze(1)
        yb = yb.squeeze(1)

        out = model(mix_x.unsqueeze(1))
        loss = mixup_criterion(criterion, out, ya, yb, lam)
        return loss
    else:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        y = y.squeeze(1)

        out = model(x.unsqueeze(1))
        loss = criterion(out, y)

        # label smoothing（与 train_Emodb.py 逻辑一致）
        lsce = LabelSmoothingCrossEntropy(smoothing=0.2)
        loss = loss + lsce(out, y)
        return loss


def _maybe_replace_last_linear(model: nn.Module, num_classes: int) -> bool:
    last_name, last_layer = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name, last_layer = name, module

    if last_layer is None:
        return False

    if last_layer.out_features == num_classes:
        return True

    parts = last_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], nn.Linear(last_layer.in_features, num_classes))
    print(f"[Patch] Replaced last Linear: {last_name} ({last_layer.out_features} -> {num_classes})")
    return True


def build_criterion(train_y: np.ndarray, use_weight: bool, num_classes: int):
    if not use_weight:
        return nn.CrossEntropyLoss()

    cnt = Counter(train_y.tolist())
    nums = np.array([cnt.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    if nums.sum() == 0:
        return nn.CrossEntropyLoss()

    weight = torch.tensor(1.0 - nums / nums.sum(), dtype=torch.float32)
    if torch.cuda.is_available():
        weight = weight.cuda()
    return nn.CrossEntropyLoss(weight=weight)


def train_loop(kws):
    global train_X, train_y, val_dict, filename, target_names

    print(f'Model: {kws["model"]}')
    shape = train_X.shape[1:]  # e.g. (26, 57)
    print("Input shape:", shape)

    model = getattr(models, kws["model"])(shape=shape)
    _maybe_replace_last_linear(model, kws["num_classes"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Total Parameters: %.3fM" % parameters)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = build_criterion(train_y, kws["weight"], kws["num_classes"])

    if kws["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=kws["learning_rate"], weight_decay=1e-6)
    elif kws["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=kws["learning_rate"], momentum=0.9,
                              weight_decay=1e-4, nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=kws["learning_rate"], weight_decay=1e-6)

    from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
    if kws["lr_schedule"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
    elif kws["lr_schedule"] == "exp":
        scheduler = ExponentialLR(optimizer, 0.95)
    else:
        scheduler = StepLR(optimizer, 10000, 1)

    os.makedirs(kws["datadir"], exist_ok=True)
    fh = open(f'{kws["datadir"]}/{kws["model"]}_train.log', "a")

    maxACC = 0
    totalrunningTime = 0
    MODEL_PATH = f'{kws["datadir"]}/model_CASIA_{kws["model"]}_{filename}.pth'

    from sklearn.metrics import classification_report
    from sklearn import metrics

    for i in range(kws["epochs"]):
        startTime = time.perf_counter()
        tq = tqdm(total=len(train_y))
        model.train()
        print_loss = 0

        for _, data in enumerate(train_loader):
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            loss = get_loss(model, criterion, x, y, kws["alpha"])
            print_loss += loss.data.item() * kws["batch_size"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.update(kws["batch_size"])

        tq.close()
        if optimizer.param_groups[0]["lr"] >= kws["lr_min"]:
            scheduler.step()

        print(f'epoch: {i}, lr: {optimizer.param_groups[0]["lr"]:.6f}, loss: {print_loss / len(train_X):.6f}')
        fh.write(f'epoch: {i}, lr: {optimizer.param_groups[0]["lr"]:.6f}, loss: {print_loss / len(train_X):.6f}\n')

        endTime = time.perf_counter()
        totalrunningTime += endTime - startTime
        fh.write(f"{totalrunningTime}\n")

        # validation/test (utterance-level avg over segments)
        model.eval()
        y_true, y_pred = [], []
        for val in val_dict:
            x, y = val["X"], val["y"]
            x = torch.from_numpy(x).float()
            y_true.append(y)

            if torch.cuda.is_available():
                x = x.cuda()

            if x.size(0) == 1:
                x = torch.cat((x, x), 0)

            out = model(x.unsqueeze(1))
            pred = out.mean(dim=0)
            pred = torch.max(pred, 0)[1].cpu().numpy()
            y_pred.append(int(pred))

        report_dict = classification_report(y_true, y_pred, digits=6, target_names=target_names, output_dict=True)
        matrix = metrics.confusion_matrix(y_true, y_pred)

        WA = report_dict["accuracy"] * 100
        UA = report_dict["macro avg"]["recall"] * 100
        ACC = (WA + UA) / 2

        if maxACC < ACC:
            maxACC = ACC
            if kws["SaveModel"]:
                torch.save(model.state_dict(), MODEL_PATH)

        print("Confusion matrix:\n", matrix)
        print(f"ACC={ACC:.4f}  WA={WA:.4f}  UA={UA:.4f}")

    del model
    return maxACC


if __name__ == "__main__":
    args = parse_args()
    kws = vars(args)

    if kws["gpu"] == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif kws["gpu"] is not None and kws["gpu"] > -1:
        torch.cuda.set_device(kws["gpu"])

    setup_seed(kws["seed"])

    # filename / features pkl
    test_speakers = [s.strip() for s in kws["test_speakers"].split(",") if s.strip()]
    if len(test_speakers) == 0:
        spk_tag = f"random80_20_seed{kws['seed']}"
    else:
        spk_tag = "test_" + "_".join(test_speakers)

    filename = f'{kws["features_to_use"]}_CASIA_{spk_tag}_hop{kws["hop_length"]}'

    if kws["featuresFileName"] is None:
        kws["featuresFileName"] = f'{kws["datadir"]}/features_{filename}.pkl'

    os.makedirs(kws["datadir"], exist_ok=True)

    with open(kws["datadir"] + "/last_casia.json", "w") as f:
        json.dump({k: v for k, v in kws.items()}, f, indent=4)

    print("Load features...")
    if kws["loadFeatures"] and os.path.exists(kws["featuresFileName"]):
        with open(kws["featuresFileName"], "rb") as f:
            features = pickle.load(f)
        train_X, train_y, val_dict, info = (
            features["train_X"],
            features["train_y"],
            features["val_dict"],
            features.get("info", ""),
        )
        if isinstance(info, dict):
            print("[Info]", info.get("mfcc_hop_length"), info.get("mfcc_target_frames"))
    else:
        kws2 = dict(kws)
        kws2.pop("test_speakers", None)

        train_X, train_y, val_dict, info = process_CASIA(
            kws2["dataset_root"],
            CASIA_LABEL,
            **kws2,
            test_speakers=(test_speakers if len(test_speakers) > 0 else None),
        )

    train_data = data_loader.DataSet(train_X, train_y)
    train_loader = DataLoader(train_data, batch_size=kws["batch_size"], shuffle=True, num_workers=0)

    target_names = ["Neutral", "Sad", "Angry", "Happy", "Fear", "Surprise"]

    best = [0.0]
    t0 = time.perf_counter()
    for _ in range(kws["iter"]):
        best_ = train_loop(kws)
        if best_ > best[0]:
            best[0] = best_

    print(f"Best ACC: {best[0]:.4f}")
    print(f"Total running time: {time.perf_counter() - t0:.2f}s")
