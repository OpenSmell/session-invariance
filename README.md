# Session-Invariance Proof for Electronic Nose Latent Spaces

[![OpenSmell](https://img.shields.io/badge/OpenSmell-project-blue)](https://opensmell.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

We prove that a neural network can learn a **session-invariant latent space** from raw electronic nose data. Using the [SmellNet](https://github.com/MIT-MI/SmellNet) dataset.

Two models are provided:

| Script | Input | Architecture | Test accuracy |
|--------|-------|-------------|---------------|
| `smellnet_latent_proof_baseline.py` | 12-dim mean+std per sensor | MLP (128→64→32) | ~75% |
| `smellnet_latent_proof_cnn.py` | Raw time series (100,6) | 1D-CNN (conv→pool→conv→pool→avgpool) | ~82% |

The baseline computes summary statistics per recording, throwing away temporal dynamics.  The 1D-CNN reads the full sensor waveform — adsorption peak, decay curve, volatility signatures — giving a more complete representation and higher accuracy on held-out sessions.

## Quick start

```bash
# 1. Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn

# 2. Run the baseline MLP (2–3 min on CPU)
python smellnet_latent_proof_baseline.py

# 3. Run the 1D-CNN on raw time series (5–10 min on CPU)
python smellnet_latent_proof_cnn.py
```

Both scripts are **fully self-contained**:
- They locate the SmellNet CSV files automatically from the HuggingFace cache
- All outputs (model weights, t-SNE plot, report) are written to the script's own directory

No need to download SmellNet separately — the first run will cache it
from HuggingFace (`DeweiFeng/smell-net`, `base_data` config).

## Dataset

We use **SmellNet** (Dewei Feng et al., MIT Media Lab), loaded via
HuggingFace.  The `base_data` config contains 6-channel time series
(NO₂, C₂H₅OH, VOC, CO, Alcohol, LPG) for 50 base substances, each with
multiple recording sessions on different days.

## How the session-invariance test works

1. For each substance, hold out the **last 2 recording sessions** as test data
   — these are measurements taken on different days, so sensor conditions
   differ.
2. Train the model on the remaining sessions.
3. If the model achieves **>70% accuracy** on the held-out sessions
   (random chance = 2%), the latent bottleneck has learned a representation
   that is invariant to session-to-session variation.

## Results

| Model | Test accuracy | Macro F1 | Success (>70%) |
|-------|-------------|----------|----------------|
| MLP baseline (12-dim stats) | 74.84% | 72.28% | ✓ |
| 1D-CNN (raw time series) | **81.78%** | **80.33%** | ✓ |

## Outputs

After running each script, you'll find:

- `model.pth` / `model_cnn.pth` — trained model weights
- `latent_tsne.png` / `latent_tsne_cnn.png` — t-SNE of the latent space (test set)
- `report.txt` / `report_cnn.txt` — accuracy, F1, and interpretation

## No GC-MS data used

All experiments rely solely on raw sensor readings and substance labels.
No chemistry data (GC-MS) is used.

## Contribute

- Try running `smellnet_latent_proof_transformer.py` (the Transformer version — may require a GPU to train in reasonable time). Make any necessary tweaks and report your results.

- Any other creative feedback or improvements you can make
