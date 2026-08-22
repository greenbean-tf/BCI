"""
Standalone 訓練腳本：使用 0521_tsai 的全部資料。

資料切法：
  - 每個 trail (1500 samples = 3s @ 500Hz) 切成 3 個不重疊的 segment (各 500 samples)
  - Trail-level split: 先在 trial 層級做 train/valid 分割（避免 data leakage）
  - 然後對各 trial 的 3 個 segment 分別做 demean + bandpass

訓練：
  - 100 epochs, patience=50, batch_size=8, lr=1e-5
  - EEGNet_our_ver (from scratch, 不做 fine-tune)

輸出：
  - valid_acc / loss 圖表 → training_result.png

用法:
    cd d:\\2026_BCI_-\\standalone_train_tsai
    python run_train.py
"""
import os
import sys
import glob
import numpy as np
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── 確保 import 本地模組 ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import preprocess as pps
from eeg_model import EEGNet_our_ver

# ================================================================
# 路徑設定
# ================================================================
# 0521_tsai 和 0521_tsai_2 的 real_time_data
DATA_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "combine_all_new", "account", "0521_tsai", "real_time_data"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "combine_all_new", "account", "0521_tsai_2", "real_time_data"),
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================================================================
# 超參數
# ================================================================
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-5
PATIENCE = 50
SEED = 42
VALID_RATIO = 0.25
TRAIL_SIZE = 1500       # 3s @ 500Hz
SEGMENT_LEN = 500       # 1s @ 500Hz
N_SEGMENTS = TRAIL_SIZE // SEGMENT_LEN  # = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ================================================================
# 1. CSV → 切 trail
# ================================================================
def csv_to_trails(csv_path):
    """讀取 CSV，切出所有有效的 trail (每個 trail = 1500 samples)"""
    DATA_pd = pd.read_csv(csv_path, skiprows=10, encoding="utf-8")

    DATA_np_x = DATA_pd[config.choose_channels_names].to_numpy()
    DATA_np_y = DATA_pd['Beatmap_Marker'].to_numpy()

    change_points = np.where(np.diff(DATA_np_y.astype(int)) != 0)[0] + 1
    split_indices = np.concatenate((change_points, [len(DATA_np_y)]))

    X_record, y_record = [], []

    for i in range(len(split_indices) - 1):
        ed = split_indices[i]
        st = ed - TRAIL_SIZE

        if DATA_np_y[ed - 1] in config.choose_marker:
            if st >= 0:
                X_record.append(DATA_np_x[st:ed].T)  # (C, T) = (13, 1500)
                y_record.append(DATA_np_y[ed - 1])
            else:
                print(f"  [Skip] trail too short: ed={ed}, st={st}")

    if len(X_record) == 0:
        return np.array([]), np.array([])

    return np.stack(X_record, axis=0), np.array(y_record)


# ================================================================
# 2. 載入所有 CSV 資料
# ================================================================
print("=" * 60)
print("  Standalone Training: 0521_tsai (No-Overlap, 3 Segments)")
print("=" * 60)

all_trails_x = []  # list of (C, 1500)
all_trails_y = []  # list of int

for data_dir in DATA_DIRS:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"\n[INFO] Data dir: {os.path.basename(os.path.dirname(data_dir))}")
    print(f"[INFO] Found {len(csv_files)} CSV files")

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        x, y = csv_to_trails(csv_path)
        if len(x) > 0:
            all_trails_x.append(x)
            all_trails_y.append(y)
            print(f"  {fname}: {len(y)} trails, classes={np.unique(y)}")
        else:
            print(f"  {fname}: no valid trails")

# 合併所有 trail
all_x = np.concatenate(all_trails_x, axis=0)  # (N_total, C, 1500)
all_y = np.concatenate(all_trails_y, axis=0)   # (N_total,)

print(f"\n[INFO] Total trails: {len(all_y)}")
print(f"[INFO] Trail shape: {all_x.shape}")
print(f"[INFO] Class distribution: ", end="")
for c in sorted(np.unique(all_y)):
    print(f"class_{int(c)}={np.sum(all_y == c)} ", end="")
print()

# ================================================================
# 3. Trial-level train/valid split（防止 data leakage）
# ================================================================
indices = np.arange(len(all_y))
train_idx, valid_idx = train_test_split(
    indices, test_size=VALID_RATIO, stratify=all_y, random_state=SEED
)

print(f"\n[INFO] Train trials: {len(train_idx)}")
print(f"[INFO] Valid trials: {len(valid_idx)}")

# ================================================================
# 4. 每個 trail 切成 3 個不重疊 segment + 前處理
# ================================================================
def split_trail_to_segments(trail_x, trail_y):
    """
    trail_x: (C, 1500) 一個 trail
    trail_y: int 標籤
    
    回傳:
        segments: list of (samples, C) = (500, 13) 已做 demean + bandpass
        labels:   list of int
    """
    segments = []
    labels = []
    for seg_i in range(N_SEGMENTS):
        start = seg_i * SEGMENT_LEN
        end = start + SEGMENT_LEN
        seg = trail_x[:, start:end].T  # (500, 13)

        # 前處理：demean + bandpass
        seg = pps.each_channel_demean(seg, axis=0)
        seg = pps.bandpass(seg)

        segments.append(seg)
        labels.append(trail_y)

    return segments, labels


train_segments, train_labels = [], []
valid_segments, valid_labels = [], []

for idx in train_idx:
    segs, labs = split_trail_to_segments(all_x[idx], all_y[idx])
    train_segments.extend(segs)
    train_labels.extend(labs)

for idx in valid_idx:
    segs, labs = split_trail_to_segments(all_x[idx], all_y[idx])
    valid_segments.extend(segs)
    valid_labels.extend(labs)

print(f"\n[INFO] Train segments (after 3-split): {len(train_segments)}")
print(f"[INFO] Valid segments (after 3-split): {len(valid_segments)}")

# ================================================================
# 5. 建構 TensorDataset
# ================================================================
def to_dataset(segment_list, label_list):
    # segment_list: list of (500, 13)
    data_x = np.transpose(np.stack(segment_list), (0, 2, 1))  # (N, C, T)
    X = torch.tensor(data_x, dtype=torch.float32).unsqueeze(1)  # (N, 1, C, T)
    y = F.one_hot(torch.tensor(label_list, dtype=torch.long),
                  num_classes=config.N_Class).float()
    return TensorDataset(X, y)


dataset_train = to_dataset(train_segments, train_labels)
dataset_valid = to_dataset(valid_segments, valid_labels)

sample_x, sample_y = dataset_train[0]
print(f"[INFO] Sample X shape: {sample_x.shape}")
print(f"[INFO] Sample Y shape: {sample_y.shape}")

# ================================================================
# 6. 訓練
# ================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[INFO] Device: {device}")

# 模型參數
n_chans = sample_x.shape[1]   # 13
n_times = sample_x.shape[2]   # 500
n_outputs = config.N_Class     # 4

model = EEGNet_our_ver(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))

train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False)

total_params = sum(p.numel() for p in model.parameters())
print(f"[INFO] Model params: {total_params:,}")
print(f"[INFO] Training for {EPOCHS} epochs (patience={PATIENCE})...\n")

history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
early_stop_counter = 0

epochBar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
for epoch in epochBar:
    # ── Train ──
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs.squeeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.argmax(dim=1)).sum().item()

    train_loss = running_loss / len(dataset_train)
    train_acc = correct / total

    # ── Evaluate ──
    model.eval()
    val_correct, val_total, val_losses = 0, 0, []

    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            output = model(x_batch.squeeze(1))
            vloss = criterion(output, y_batch)
            val_losses.append(vloss.item())

            preds = output.argmax(dim=1)
            val_correct += (preds == y_batch.argmax(dim=1)).sum().item()
            val_total += y_batch.size(0)

    val_loss = np.mean(val_losses)
    val_acc = val_correct / val_total

    history['loss'].append(train_loss)
    history['acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, os.path.join(CHECKPOINT_DIR, f"train-epoch{epoch}.pth"))

    epochBar.set_postfix(
        loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}",
        val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}"
    )

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}. Best val_loss: {best_val_loss:.4f}")
        break

# ================================================================
# 7. 結果
# ================================================================
best_loss_epoch = int(np.argmin(history['val_loss']))
best_val_acc = history['val_acc'][best_loss_epoch]
best_val_loss = history['val_loss'][best_loss_epoch]

print(f"\n{'=' * 60}")
print(f"  RESULT (No-Overlap, 3 Segments per Trail)")
print(f"{'=' * 60}")
print(f"  Total epochs trained : {len(history['val_acc'])}")
print(f"  Best epoch (min val_loss): {best_loss_epoch + 1}")
print(f"  Best val_acc  = {best_val_acc:.4f}")
print(f"  Best val_loss = {best_val_loss:.4f}")
print(f"  Final train_acc = {history['acc'][-1]:.4f}")
print(f"  Final val_acc   = {history['val_acc'][-1]:.4f}")
print(f"{'=' * 60}")

# ================================================================
# 8. 繪製圖表
# ================================================================
epochs_range = range(1, len(history['val_acc']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('0521_TSAI — No-Overlap 3-Segment Training (100 epochs)',
             fontsize=16, fontweight='bold')

# ── Accuracy ──
ax1 = axes[0]
ax1.plot(epochs_range, history['acc'], label='Train Acc', color='#4A90D9', linewidth=2)
ax1.plot(epochs_range, history['val_acc'], label='Valid Acc', color='#E74C3C', linewidth=2)
ax1.axvline(x=best_loss_epoch + 1, color='gray', linestyle='--', alpha=0.6,
            label=f'Best epoch = {best_loss_epoch + 1}')
ax1.scatter([best_loss_epoch + 1], [best_val_acc], color='#E74C3C', s=100, zorder=5,
            edgecolors='black', linewidths=1.5)
ax1.annotate(f'{best_val_acc:.4f}',
             xy=(best_loss_epoch + 1, best_val_acc),
             xytext=(best_loss_epoch + 5, best_val_acc - 0.05),
             fontsize=11, fontweight='bold', color='#E74C3C',
             arrowprops=dict(arrowstyle='->', color='#E74C3C'))
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy Curve', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ── Loss ──
ax2 = axes[1]
ax2.plot(epochs_range, history['loss'], label='Train Loss', color='#4A90D9', linewidth=2)
ax2.plot(epochs_range, history['val_loss'], label='Valid Loss', color='#E74C3C', linewidth=2)
ax2.axvline(x=best_loss_epoch + 1, color='gray', linestyle='--', alpha=0.6,
            label=f'Best epoch = {best_loss_epoch + 1}')
ax2.scatter([best_loss_epoch + 1], [best_val_loss], color='#E74C3C', s=100, zorder=5,
            edgecolors='black', linewidths=1.5)
ax2.annotate(f'{best_val_loss:.4f}',
             xy=(best_loss_epoch + 1, best_val_loss),
             xytext=(best_loss_epoch + 5, best_val_loss + 0.05),
             fontsize=11, fontweight='bold', color='#E74C3C',
             arrowprops=dict(arrowstyle='->', color='#E74C3C'))
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Loss Curve', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "training_result.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[INFO] Chart saved to: {output_path}")

# 也存一份 history 供後續分析
import json
hist_path = os.path.join(OUTPUT_DIR, "training_history.json")
with open(hist_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"[INFO] History saved to: {hist_path}")

print("\nDone!")
