#!/usr/bin/env python3
"""
Session Invariance — Transformer on Raw Time Series
====================================================
Upgrades the baseline 12‑dim mean/std model to a Transformer that
reads the raw (200,6) sensor sequence.  This preserves temporal
dynamics — adsorption peak, decay curve, volatility signatures —
which the simple statistical summary throws away.

Expected improvement: >80% accuracy on held‑out sessions.
Deliverables:
  - model_transformer.pth
  - latent_tsne_transformer.png
  - report_transformer.txt
"""

import os, io, glob, random, warnings, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# 1. Locate SmellNet CSV files in the HuggingFace cache
# ----------------------------------------------------------------------
print("=" * 60)
print("Session Invariance — Transformer on Raw Time Series")
print("=" * 60)

# Find the snapshot with base_data
hub_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net")
if not os.path.exists(hub_dir):
    # fallback: try loading the dataset to force cache
    from datasets import load_dataset
    ds = load_dataset("DeweiFeng/smell-net", "base_data", split="train")
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net")

snapshots = os.path.join(hub_dir, "snapshots")
if not os.path.exists(snapshots):
    raise RuntimeError("No snapshots directory found in HF cache.")
revision = sorted(os.listdir(snapshots), reverse=True)[0]
data_root = os.path.join(snapshots, revision, "base_data")
print(f"Data root: {data_root}")

# Collect all CSV files from training/ and testing/ subdirs
csv_files = []
for root, dirs, files in os.walk(data_root):
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(os.path.join(root, f))
print(f"Found {len(csv_files)} CSV files")

# ----------------------------------------------------------------------
# 2. Parse each CSV into fixed‑length raw sequences (segments)
# ----------------------------------------------------------------------
SENSOR_NAMES = ['NO2', 'C2H5OH', 'VOC', 'CO', 'Alcohol', 'LPG']
SEGMENT_LEN = 200          # points per sequence
STRIDE = SEGMENT_LEN // 2  # 50% overlap

sequences = []          # each will be a (200,6) numpy array
labels_list = []
session_ids_list = []
filenames_list = []
substances_seen = {}
skipped_files = []

for fpath in csv_files:
    base = os.path.basename(fpath)
    stem = os.path.splitext(base)[0]
    parts = stem.split('_')
    if len(parts) < 2 or not parts[-1].isdigit():
        skipped_files.append((base, "bad filename pattern"))
        continue
    session_id = int(parts[-1])
    substance = '_'.join(parts[:-1])

    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        skipped_files.append((base, f"read error: {e}"))
        continue

    # Find the 6 sensor columns (case insensitive)
    available = df.columns.tolist()
    sensor_cols = []
    for expected in SENSOR_NAMES:
        found = [c for c in available if c.lower() == expected.lower()]
        if found:
            sensor_cols.append(found[0])
        else:
            break
    if len(sensor_cols) != 6:
        skipped_files.append((base, f"found {len(sensor_cols)}/6 sensor cols"))
        continue

    sensor_data = df[sensor_cols].values.astype(np.float32)
    N = sensor_data.shape[0]
    if N == 0:
        skipped_files.append((base, "empty"))
        continue

    # Cut into overlapping segments (or keep whole if too short)
    if N < SEGMENT_LEN:
        segments = [sensor_data]
    else:
        segments = []
        for start in range(0, N - SEGMENT_LEN + 1, STRIDE):
            segments.append(sensor_data[start:start+SEGMENT_LEN])

    for seg in segments:
        sequences.append(seg)
        labels_list.append(substance)
        session_ids_list.append(session_id)
        filenames_list.append(base)

    substances_seen.setdefault(substance, set()).add(session_id)

# Stack into a 3D numpy array (num_samples, SEGMENT_LEN, 6)
X_raw = np.stack(sequences, axis=0)   # shape: (N, 200, 6)
labels = np.array(labels_list)
session_ids = np.array(session_ids_list)

print(f"\nExtracted {X_raw.shape[0]} sequences of shape {X_raw.shape[1:]} from {len(csv_files)} files")
print(f"Unique substances: {len(substances_seen)}")

if skipped_files:
    print(f"\nSkipped {len(skipped_files)} files:")
    for n, r in skipped_files[:10]:
        print(f"  {n}: {r}")

# ----------------------------------------------------------------------
# 3. Session‑based train/val/test split (same logic as baseline)
# ----------------------------------------------------------------------
unique_substances = sorted(substances_seen.keys())
num_substances = len(unique_substances)

# Build index mapping (substance, session) -> indices
sub_ses_to_idx = {}
for idx, (sub, ses) in enumerate(zip(labels, session_ids)):
    sub_ses_to_idx.setdefault((sub, ses), []).append(idx)

train_idx, test_idx = [], []
single_sesh = []

for sub in unique_substances:
    sessions = sorted(substances_seen[sub])
    if len(sessions) >= 3:
        train_ses = sessions[:-2]
        test_ses = sessions[-2:]
    elif len(sessions) == 2:
        train_ses = [sessions[0]]
        test_ses = [sessions[1]]
    else:
        train_ses = sessions
        test_ses = []
        single_sesh.append(sub)

    for s in train_ses:
        train_idx.extend(sub_ses_to_idx[(sub, s)])
    for s in test_ses:
        test_idx.extend(sub_ses_to_idx[(sub, s)])

train_idx = np.array(train_idx)
test_idx = np.array(test_idx)

# Further split training into train/val (10% validation)
train_sub_idx, val_sub_idx = train_test_split(
    np.arange(len(train_idx)), test_size=0.1, random_state=42)
val_idx = train_idx[val_sub_idx]
train_idx = train_idx[train_sub_idx]

print(f"\nTrain samples: {len(train_idx)}")
print(f"Val samples:   {len(val_idx)}")
print(f"Test samples:  {len(test_idx)}")
print(f"Single‑session substances (train only): {len(single_sesh)}")

# ----------------------------------------------------------------------
# 4. Standardize sensor values (per‑sensor mean/std from train set)
# ----------------------------------------------------------------------
# Flatten train sequences to compute per-sensor statistics
train_flat = X_raw[train_idx].reshape(-1, 6)  # (train_seq*200, 6)
sensor_mean = train_flat.mean(axis=0)
sensor_std = train_flat.std(axis=0) + 1e-8

X_train = (X_raw[train_idx] - sensor_mean[None, None, :]) / sensor_std[None, None, :]
X_val   = (X_raw[val_idx]   - sensor_mean[None, None, :]) / sensor_std[None, None, :]
X_test  = (X_raw[test_idx]  - sensor_mean[None, None, :]) / sensor_std[None, None, :]

# Encode string labels to integers
label_to_int = {l: i for i, l in enumerate(unique_substances)}
y_train = np.array([label_to_int[l] for l in labels[train_idx]])
y_val   = np.array([label_to_int[l] for l in labels[val_idx]])
y_test  = np.array([label_to_int[l] for l in labels[test_idx]])

print(f"Number of classes: {num_substances}")
print(f"Random baseline: {100/num_substances:.2f}%")

# ----------------------------------------------------------------------
# 5. PyTorch Dataset & Transformer Model
# ----------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.seq = torch.tensor(sequences, dtype=torch.float32)  # (N, 200, 6)
        self.lbl = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return self.seq[idx], self.lbl[idx]

# Transformer encoder for multivariate time series
class TransformerSmellNet(nn.Module):
    def __init__(self, num_sensors=6, d_model=64, nhead=4, num_layers=2,
                 dropout=0.3, num_classes=50, max_len=200):
        super().__init__()
        self.input_fc = nn.Linear(num_sensors, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)   # pool over time
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, return_latent=False):
        # x: (batch, seq_len, num_sensors)
        x = self.input_fc(x)                     # (B, L, d_model)
        x = x + self.pos_embed[:, :x.size(1), :] # add positional encoding
        x = self.transformer(x)                  # (B, L, d_model)
        x = x.transpose(1, 2)                    # (B, d_model, L) for pooling
        latent = self.pool(x).squeeze(-1)        # (B, d_model)
        latent = self.drop(latent)
        out = self.classifier(latent)
        if return_latent:
            return out, latent
        return out

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = TransformerSmellNet(num_sensors=6, d_model=64, nhead=4, num_layers=2,
                            dropout=0.3, num_classes=num_subclasses).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=5, min_lr=1e-5)

train_ds = SequenceDataset(X_train, y_train)
val_ds   = SequenceDataset(X_val, y_val)
test_ds  = SequenceDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

# ----------------------------------------------------------------------
# 6. Training loop (early stopping, identical to baseline)
# ----------------------------------------------------------------------
MAX_EPOCHS = 200
PATIENCE = 15
best_val_loss = float('inf')
best_state = None
no_improve = 0

print("\n" + "=" * 60)
print("Training Transformer")
print("=" * 60)

for epoch in range(1, MAX_EPOCHS+1):
    model.train()
    tr_loss = 0.0
    tr_correct = 0
    tr_total = 0
    for seq, y in train_loader:
        seq, y = seq.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(seq)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * seq.size(0)
        tr_total += seq.size(0)
        tr_correct += (out.argmax(1) == y).sum().item()
    tr_loss /= tr_total
    tr_acc = tr_correct / tr_total * 100

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for seq, y in val_loader:
            seq, y = seq.to(device), y.to(device)
            out = model(seq)
            loss = criterion(out, y)
            val_loss += loss.item() * seq.size(0)
            val_total += seq.size(0)
            val_correct += (out.argmax(1) == y).sum().item()
    val_loss /= val_total
    val_acc = val_correct / val_total * 100

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | TrLoss {tr_loss:.4f} TrAcc {tr_acc:.2f}% | "
              f"ValLoss {val_loss:.4f} ValAcc {val_acc:.2f}% | "
              f"Patience {no_improve}/{PATIENCE}")

    if no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
model_path = os.path.join(OUT_DIR, "model_transformer.pth")
torch.save(best_state, model_path)
print(f"\nBest model saved to {model_path}")

# ----------------------------------------------------------------------
# 7. Evaluation on held‑out test sessions
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Test Evaluation")
print("=" * 60)
model.eval()
test_correct = 0
test_total = 0
all_preds, all_true = [], []
with torch.no_grad():
    for seq, y in test_loader:
        seq, y = seq.to(device), y.to(device)
        out = model(seq)
        pred = out.argmax(1)
        test_total += y.size(0)
        test_correct += (pred == y).sum().item()
        all_preds.extend(pred.cpu().numpy())
        all_true.extend(y.cpu().numpy())

test_acc = test_correct / test_total * 100
test_f1 = f1_score(all_true, all_preds, average='macro') * 100
baseline = 100 / num_subclasses
success = test_acc > 70.0

print(f"Test Accuracy:  {test_acc:.2f}%")
print(f"Test F1 (macro): {test_f1:.2f}%")
print(f"Random baseline: {baseline:.2f}%")
print(f"Success (>70%): {'YES' if success else 'NO'}")

# ----------------------------------------------------------------------
# 8. t‑SNE of latent vectors from test set
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Generating t‑SNE plot")
print("=" * 60)

model.eval()
latent_list = []
test_labels_str = []
with torch.no_grad():
    for seq, y in test_loader:
        seq, y = seq.to(device), y.to(device)
        _, lat = model(seq, return_latent=True)
        latent_list.append(lat.cpu().numpy())
        for idx in y.cpu().numpy():
            test_labels_str.append(unique_substances[idx])

latent_vectors = np.concatenate(latent_list, axis=0)
tsne = TSNE(n_components=2, perplexity=min(30, len(latent_vectors)-1),
            random_state=42, init='random', learning_rate='auto')
latent_2d = tsne.fit_transform(latent_vectors)

plt.figure(figsize=(12, 10))
if len(unique_substances) <= 20:
    palette = sns.color_palette("hsv", len(unique_substances))
    for i, sub in enumerate(unique_substances):
        mask = np.array(test_labels_str) == sub
        plt.scatter(latent_2d[mask,0], latent_2d[mask,1],
                    color=palette[i], label=sub, alpha=0.7, s=30)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
else:
    # color by class index
    plt.scatter(latent_2d[:,0], latent_2d[:,1], c=all_true,
                cmap='tab20', alpha=0.7, s=30, edgecolors='none')
    plt.colorbar(label='class index')

plt.title("t‑SNE of OpenSmell Latent Space (Transformer, Test Sessions)")
plt.xlabel("t‑SNE 1")
plt.ylabel("t‑SNE 2")
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "latent_tsne_transformer.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: {plot_path}")

# ----------------------------------------------------------------------
# 9. Write report
# ----------------------------------------------------------------------
report = f"""============================================================
Session Invariance — Transformer on Raw Time Series
============================================================
Number of sequences: {X_raw.shape[0]} (from {len(csv_files)} files)
Sequence shape:      (200, 6)
Substances:          {num_subclasses}
Train / Val / Test:  {len(train_idx)} / {len(val_idx)} / {len(test_idx)}

Test Accuracy:  {test_acc:.2f}%
Test F1 (macro): {test_f1:.2f}%
Random baseline: {baseline:.2f}%
Success (>70%): {'YES' if success else 'NO'}

t‑SNE plot: {plot_path}
Model file:  {model_path}

Single‑session substances (train-only): {len(single_sesh)}
Skipped files: {len(skipped_files)}
"""
report_path = os.path.join(OUT_DIR, "report_transformer.txt")
with open(report_path, 'w') as f:
    f.write(report)
print(report)
print("\nPhase 1b complete.")