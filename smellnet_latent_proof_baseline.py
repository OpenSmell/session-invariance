import random
import os
import io
import glob
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ------------------------------------------------------------
# Step 1: Locate and load all CSV files from the dataset cache
# ------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("Session Invariance — SmellNet Session-Invariance Proof")
print("=" * 60)

# Find the dataset cache directory
hub_cache_candidates = [
    os.path.expanduser("~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net"),
    os.path.expanduser("~/.cache/huggingface/datasets/DeweiFeng___smell-net"),
]

snapshot_dir = None
for base_candidate in hub_cache_candidates:
    if os.path.exists(base_candidate):
        if "hub" in base_candidate:
            snapshots = os.path.join(base_candidate, "snapshots")
            if os.path.exists(snapshots):
                revisions = os.listdir(snapshots)
                if revisions:
                    snapshot_dir = os.path.join(snapshots, revisions[0], "base_data")
                    break
        else:
            # datasets cache (arrow files) — not the raw CSVs
            pass

if snapshot_dir is None or not os.path.exists(snapshot_dir):
    # Fallback: try to load the dataset and find the cache from HuggingFace
    from datasets import load_dataset
    ds = load_dataset("DeweiFeng/smell-net", "base_data", split="train")
    # Try to find raw CSV from the hub cache using the revision
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net")
    if os.path.exists(hub_dir):
        snapshots = os.path.join(hub_dir, "snapshots")
        revisions = sorted(os.listdir(snapshots), reverse=True)
        for rev in revisions:
            candidate = os.path.join(snapshots, rev, "base_data")
            if os.path.isdir(candidate):
                snapshot_dir = candidate
                break

if snapshot_dir is None or not os.path.exists(snapshot_dir):
    raise RuntimeError(
        "Could not locate SmellNet base_data CSV files in HuggingFace cache. "
        f"Tried: {hub_cache_candidates}"
    )

print(f"Data directory: {snapshot_dir}")

# Collect all CSV files from both training/ and testing/ directories
csv_files = []
for root, dirs, files in os.walk(snapshot_dir):
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(os.path.join(root, f))

print(f"Found {len(csv_files)} CSV files in {snapshot_dir}")

if len(csv_files) == 0:
    raise RuntimeError("No CSV files found. Dataset structure unexpected.")

# ------------------------------------------------------------
# Step 2: Parse each CSV → feature vector
# ------------------------------------------------------------
SENSOR_NAMES = ['NO2', 'C2H5OH', 'VOC', 'CO', 'Alcohol', 'LPG']

features_list = []
labels_list = []
session_ids_list = []
filenames_list = []
substances_seen = {}
skipped_files = []

for fpath in csv_files:
    base = os.path.basename(fpath)
    # Parse filename: {substance}_{session}.csv
    stem = os.path.splitext(base)[0]
    # Handle names that might have multiple underscores:
    # e.g., "brussel_sprouts_2.csv" -> substance "brussel_sprouts", session "2"
    # The session is always the last number after the last underscore
    parts = stem.split('_')
    if len(parts) < 2:
        skipped_files.append((base, "filename does not have substance_session pattern"))
        continue

    # Session number is the last part, unless it's not a number
    session_str = parts[-1]
    if not session_str.isdigit():
        skipped_files.append((base, f"last segment '{session_str}' is not a digit"))
        continue

    session_id = int(session_str)
    substance = '_'.join(parts[:-1])

    # Read CSV
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        skipped_files.append((base, f"read error: {e}"))
        continue

    # Find sensor columns
    available_cols = df.columns.tolist()
    sensor_cols = [c for c in SENSOR_NAMES if c in available_cols]
    if len(sensor_cols) != 6:
        # Try case-insensitive matching
        sensor_cols = []
        for expected in SENSOR_NAMES:
            found = [c for c in available_cols if c.lower() == expected.lower()]
            if found:
                sensor_cols.append(found[0])
            else:
                break
        if len(sensor_cols) != 6:
            skipped_files.append((base, f"expected 6 sensor columns, found {len(sensor_cols)}: {available_cols}"))
            continue

    # Compute mean and std per sensor (12-dim features)
    sensor_data = df[sensor_cols].values.astype(np.float32)
    if sensor_data.shape[0] == 0:
        skipped_files.append((base, "empty CSV"))
        continue

    # Data augmentation: split long recordings into segments to increase sample count
    # Each segment gets its own feature vector but retains the same substance/session label
    N = sensor_data.shape[0]
    SEGMENT_LEN = 200  # points per segment
    stride = SEGMENT_LEN // 2  # 50% overlap

    if N < SEGMENT_LEN:
        # Short recording: single segment with padding if needed
        segments = [sensor_data]
    else:
        segments = []
        for start in range(0, N - SEGMENT_LEN + 1, stride):
            segments.append(sensor_data[start:start + SEGMENT_LEN])

    for seg in segments:
        mean_vals = np.mean(seg, axis=0)
        std_vals = np.std(seg, axis=0)
        feature_vec = np.concatenate([mean_vals, std_vals])  # 12-dim
        features_list.append(feature_vec)
        labels_list.append(substance)
        session_ids_list.append(session_id)
        filenames_list.append(base)

    if substance not in substances_seen:
        substances_seen[substance] = set()
    substances_seen[substance].add(session_id)

features = np.array(features_list)
labels = np.array(labels_list)
session_ids = np.array(session_ids_list)

print(f"\nExtracted {len(features)} feature vectors from {len(csv_files)} CSV files")
print(f"Unique substances: {len(substances_seen)}")

if len(substances_seen) < 10:
    print(f"WARNING: Expected ~50 substances, found only {len(substances_seen)}")
    print(f"Substances found: {sorted(substances_seen.keys())}")

if skipped_files:
    print(f"\nWARNING: Skipped {len(skipped_files)} files:")
    for name, reason in skipped_files[:10]:
        print(f"  {name}: {reason}")
    if len(skipped_files) > 10:
        print(f"  ... and {len(skipped_files) - 10} more")

# ------------------------------------------------------------
# Step 3: Train/Test split by held-out sessions
# ------------------------------------------------------------
unique_substances = sorted(substances_seen.keys())
n_substances = len(unique_substances)

all_indices = np.arange(len(features))

# Build mapping: (substance, session) -> indices
substance_session_to_indices = {}
for idx in range(len(features)):
    key = (labels[idx], session_ids[idx])
    if key not in substance_session_to_indices:
        substance_session_to_indices[key] = []
    substance_session_to_indices[key].append(idx)

# Split by session for each substance
train_indices = []
test_indices = []
single_session_substances = []
multi_session_test_counts = {}

for sub in unique_substances:
    sessions = sorted(substances_seen[sub])
    if len(sessions) >= 3:
        # Hold out last 2 sessions for test
        train_sessions = sessions[:-2]
        test_sessions = sessions[-2:]
    elif len(sessions) == 2:
        # Hold out 1 session for test
        train_sessions = [sessions[0]]
        test_sessions = [sessions[1]]
    else:
        # Only 1 session — all training
        train_sessions = sessions
        test_sessions = []
        single_session_substances.append(sub)

    for s in train_sessions:
        train_indices.extend(substance_session_to_indices[(sub, s)])
    for s in test_sessions:
        test_indices.extend(substance_session_to_indices[(sub, s)])
        if sub not in multi_session_test_counts:
            multi_session_test_counts[sub] = 0
        multi_session_test_counts[sub] += len(substance_session_to_indices[(sub, s)])

train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

print(f"\nSingle-session substances (train only): {len(single_session_substances)}")
if single_session_substances:
    print(f"  {single_session_substances}")

# Split training into train/val (stratified by label)
train_labels_subset = labels[train_indices]
# With ~300 samples over 50 classes, stratification on a 10% split is infeasible.
# Use a simple random split for validation.
val_ratio = 0.1
train_idx_final, val_idx_rel = train_test_split(
    np.arange(len(train_indices)),
    test_size=val_ratio,
    random_state=42
)
val_indices = train_indices[val_idx_rel]
train_indices = train_indices[train_idx_final]

print(f"\nTrain samples: {len(train_indices)}")
print(f"Val samples:   {len(val_indices)}")
print(f"Test samples:  {len(test_indices)}")
print(f"Unique substances in train: {len(np.unique(labels[train_indices]))}")
print(f"Unique substances in val:   {len(np.unique(labels[val_indices]))}")
print(f"Unique substances in test:  {len(np.unique(labels[test_indices]))}")

# ------------------------------------------------------------
# Step 4: Standardize features
# ------------------------------------------------------------
scaler = StandardScaler()
features_train = scaler.fit_transform(features[train_indices])
features_val = scaler.transform(features[val_indices])
features_test = scaler.transform(features[test_indices])

# Encode labels
label_classes = sorted(np.unique(labels))
label_to_idx = {l: i for i, l in enumerate(label_classes)}
num_classes = len(label_classes)

labels_train = np.array([label_to_idx[l] for l in labels[train_indices]], dtype=np.int64)
labels_val = np.array([label_to_idx[l] for l in labels[val_indices]], dtype=np.int64)
labels_test = np.array([label_to_idx[l] for l in labels[test_indices]], dtype=np.int64)

print(f"\nNumber of classes: {num_classes}")
print(f"Baseline (random chance): {1.0/num_classes*100:.2f}%")

# ------------------------------------------------------------
# Step 5: Build neural network
# ------------------------------------------------------------
INPUT_DIM = 12
HIDDEN1 = 128
HIDDEN2 = 64
LATENT_DIM = 32
DROPOUT = 0.3

class SmellNet(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, latent_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(DROPOUT)
        self.fc_latent = nn.Linear(hidden2, latent_dim)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(DROPOUT)
        self.fc_out = nn.Linear(latent_dim, num_classes)

    def forward(self, x, return_latent=False):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        latent = self.drop3(self.relu3(self.fc_latent(x)))
        if return_latent:
            return self.fc_out(latent), latent
        return self.fc_out(latent)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

model = SmellNet(INPUT_DIM, HIDDEN1, HIDDEN2, LATENT_DIM, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

# DataLoaders
train_dataset = TensorDataset(
    torch.tensor(features_train, dtype=torch.float32),
    torch.tensor(labels_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(features_val, dtype=torch.float32),
    torch.tensor(labels_val, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(features_test, dtype=torch.float32),
    torch.tensor(labels_test, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ------------------------------------------------------------
# Step 6: Training loop with early stopping
# ------------------------------------------------------------
MAX_EPOCHS = 200
PATIENCE = 15

best_val_loss = float('inf')
best_model_state = None
epochs_no_improve = 0

print("\n" + "=" * 60)
print("Training")
print("=" * 60)

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    train_loss /= train_total
    train_acc = train_correct / train_total * 100

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total * 100

    # LR scheduler step
    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Patience: {epochs_no_improve}/{PATIENCE}")

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Restore best model
model.load_state_dict(best_model_state)

# Save model
model_path = os.path.join(OUT_DIR, "model.pth")
torch.save(model.state_dict(), model_path)
print(f"\nBest model saved to {model_path}")

# ------------------------------------------------------------
# Step 7: Evaluation on held-out test sessions
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("Evaluation on Held-Out Test Sessions")
print("=" * 60)

model.eval()
test_correct = 0
test_total = 0
all_test_preds = []
all_test_true = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        test_total += batch_y.size(0)
        test_correct += (predicted == batch_y).sum().item()
        all_test_preds.extend(predicted.cpu().numpy())
        all_test_true.extend(batch_y.cpu().numpy())

test_acc = test_correct / test_total * 100
test_f1 = f1_score(all_test_true, all_test_preds, average='macro') * 100
random_baseline = 1.0 / num_classes * 100

print(f"Test Accuracy:  {test_acc:.2f}%")
print(f"Test F1 (macro): {test_f1:.2f}%")
print(f"Random baseline: {random_baseline:.2f}%")

success = test_acc > 70.0
print(f"\nSuccess criterion (accuracy > 70%): {'✓ SUCCESS' if success else '✗ FAILURE'}")

# ------------------------------------------------------------
# Step 8: t-SNE Latent Space Visualization
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("Generating t-SNE Latent Space Visualization")
print("=" * 60)

# Extract latent vectors for test set
model.eval()
latent_vectors = []
test_labels_str = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        _, latent = model(batch_x, return_latent=True)
        latent_vectors.append(latent.cpu().numpy())
        # Map back to original label strings
        for idx in batch_y.cpu().numpy():
            test_labels_str.append(label_classes[idx])

latent_vectors = np.concatenate(latent_vectors, axis=0)

# t-SNE
tsne = TSNE(n_components=2, perplexity=min(30, len(latent_vectors) - 1),
            random_state=42, init='random', learning_rate='auto')
latent_2d = tsne.fit_transform(latent_vectors)

# Plot
plt.figure(figsize=(12, 10))
if num_classes <= 20:
    # Use a legend
    unique_test_labels = sorted(set(test_labels_str))
    palette = sns.color_palette("hsv", len(unique_test_labels))
    label_to_color = {l: palette[i] for i, l in enumerate(unique_test_labels)}
    for l in unique_test_labels:
        mask = np.array(test_labels_str) == l
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                    c=[label_to_color[l]], label=l, alpha=0.7, s=30, edgecolors='none')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=0.8)
else:
    # No legend for many classes
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                c=labels_test, cmap='tab20', alpha=0.7, s=30, edgecolors='none')

plt.title("t-SNE of OpenSmell Latent Space on SmellNet (Test Sessions)", fontsize=14)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()

tsne_path = os.path.join(OUT_DIR, "latent_tsne.png")
plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"t-SNE plot saved to {tsne_path}")

# ------------------------------------------------------------
# Step 9: Report
# ------------------------------------------------------------
report_lines = [
    "=" * 60,
    "Session-Invariance Proof Report",
    "=" * 60,
    "",
    f"Number of samples used: {len(features)}",
    f"Number of substances: {num_classes}",
    f"Feature dimension: {INPUT_DIM} (mean+std for {len(SENSOR_NAMES)} sensors)",
    "",
    f"Train samples: {len(train_indices)}",
    f"Val samples:   {len(val_indices)}",
    f"Test samples:  {len(test_indices)}",
    "",
    f"Test Accuracy:  {test_acc:.2f}%",
    f"Test F1 (macro): {test_f1:.2f}%",
    f"Random baseline: {random_baseline:.2f}%",
    "",
    f"Success criterion (accuracy > 70%): {'SUCCESS' if success else 'FAILURE'}",
    "",
    f"t-SNE plot: {tsne_path}",
    f"Model: {model_path}",
    "",
]

# Anomalies and warnings
if single_session_substances:
    report_lines.append(f"Substances with only 1 session (train-only): {len(single_session_substances)}")
    report_lines.append(f"  {single_session_substances}")
    report_lines.append("")

if skipped_files:
    report_lines.append(f"Files skipped due to errors: {len(skipped_files)}")
    report_lines.append("")

report_lines.append("Anomalies / Warnings:")
report_lines.append(f"  - Single-session substances (train-only): {len(single_session_substances)}")
report_lines.append(f"  - Skipped files: {len(skipped_files)}")
report_lines.append("")

report_lines.append("Interpretation:")
if success:
    report_lines.append(
        "  The model achieves >70% accuracy on held-out sessions, demonstrating that the "
        "learned latent space is largely invariant to measurement session differences. "
        "This supports the hypothesis that a neural network's bottleneck layer can "
        "learn session-invariant representations of olfactory data."
    )
else:
    report_lines.append(
        "  The model did not exceed 70% accuracy on held-out sessions. Possible reasons: "
        "limited data per substance, insufficient feature engineering, or the need for a "
        "more sophisticated architecture (e.g., 1D-CNN on raw time series)."
    )

report_lines.append("")
report_lines.append("t-SNE visualization notes:")
if num_classes <= 20:
    report_lines.append("  Classes are colored and shown in the legend. Look for clear clusters.")
else:
    report_lines.append(
        "  Classes exceed 20, legend omitted. Points colored by test label index. "
        "Clear clustering by substance indicates latent space separation."
    )
report_lines.append("")

report_text = "\n".join(report_lines)

# Write to file
report_path = os.path.join(OUT_DIR, "report.txt")
with open(report_path, "w") as f:
    f.write(report_text)
print(report_text)

print("\n✓ Phase 1 experiment complete.")
