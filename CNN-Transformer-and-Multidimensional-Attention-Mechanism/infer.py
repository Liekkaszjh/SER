import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
import numpy as np

import models
from process_CASIA import load_wav, segment, FeatureExtractor

# 情感类别
TARGET_NAMES = ["Neutral", "Sad", "Angry", "Happy", "Fear", "Surprise"]

# ====== 修改成你的路径 ======
WAV_PATH = r"E:\SER0103\bc5e6\casia\spk1\fear\201.wav"
CKPT_PATH = r"data\CASIA\model_CASIA_CTMAM_EMODB_mfcc_CASIA_random80_20_seed2022_hop512.pth"
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ 读取音频
wav = load_wav(WAV_PATH, target_sr=16000)

# 2️⃣ 切片（和训练一致）
Xseg = segment(
    wav,
    sample_rate=16000,
    segment_length=1.8,
    overlap=1.6,
    padding=True,
)

# 3️⃣ 提取 MFCC（必须和训练一致）
extractor = FeatureExtractor(
    sample_rate=16000,
    nmfcc=26,
    hop_length=512,
    target_frames=57,
    segment_length=1.8,
)

feats = extractor.get_features("mfcc", Xseg)  # [N, 26, 57]

# 4️⃣ 构建模型
shape = feats.shape[1:]  # (26, 57)
model = models.CTMAM_EMODB(shape=shape)

# 替换最后一层为 6 类（保险起见）
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_name = name
        last_layer = module

if last_layer.out_features != 6:
    parts = last_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], torch.nn.Linear(last_layer.in_features, 6))

# 加载权重
state_dict = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 5️⃣ 推理（对所有 segment 平均）
with torch.no_grad():
    x = torch.from_numpy(feats).float().to(device)
    logits = model(x.unsqueeze(1))      # [N, 6]
    avg_logits = logits.mean(dim=0)     # [6]
    probs = F.softmax(avg_logits, dim=0).cpu().numpy()

# 6️⃣ 输出概率
print("\n=== Emotion Probabilities ===")
for name, p in zip(TARGET_NAMES, probs):
    print(f"{name:9s}: {p:.4f}")

print(f"\nPredicted Emotion: {TARGET_NAMES[int(np.argmax(probs))]}")
