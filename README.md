# Flood-Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> Official implementation for the paper accepted by **Summer 2026 Volume of Young Scholars Academic**.

---

## 📦 Overview

This repository implements a **D-GCN-LSTM-Seq2Seq model with physical constraint loss** for flood front prediction using **CFD probe data**—the virtual counterpart to the **Mobile Spherical Sensing Unit (MSSU) hardware**.

**Key features**:
- Velocity-Guided Dynamic Acyclic Graph (VGDAG) — real-time topology adaptation based on flow velocity and direction
- Sparse edge attention for efficient message passing on dynamic graphs
- Physics-informed loss (velocity bounds, spatial gradients, flow direction)
- 12-step ahead prediction (60 minutes) of 7 physical fields
- Automatic flood front centroid extraction and velocity analysis

> **For comparison**, we also provide two baseline models: pure LSTM (no spatial graph) and static GCN-LSTM (fixed KNN graph). All three models share identical optimization settings (`AdamW`, `lr=5e-5`, `weight_decay=1e-4`, gradient clipping `max_norm=1.0`) and random seed 42.

| Model | File | Role |
| :--- | :--- | :--- |
| **D-GCN-LSTM-Seq2Seq** | `d_gcn_lstm_seq2seq.py` | **Proposed model** — dynamic graph with VGDAG |
| GCN-LSTM | `gcn_lstm_seq2seq.py` | Baseline — static KNN graph |
| Pure LSTM | `lstm_seq2seq.py` | Baseline — no spatial graph |

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/Qingran-Yu/Flood-Prediction.git
cd Flood-Prediction
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .\.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

Choose the option that matches your hardware:

**Option A: CPU only (default, works on any machine)**
```bash
pip install -r requirements.txt
```

**Option B: NVIDIA GPU (CUDA 11.8)**
```bash
# Python 3.11 or lower is required for PyTorch 2.0.1
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> **Note**: If you are unsure, use Option A — it works on any machine. For GPU acceleration, ensure NVIDIA drivers and CUDA 11.8 are installed. If you encounter errors with `torch-scatter` or `torch-sparse` (PyTorch Geometric dependencies), refer to the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for platform-specific wheels.

### 4. Run training + evaluation

**Run the proposed model**:
```bash
python d_gcn_lstm_seq2seq.py
```

**Run baselines (for reproducing comparative experiments)**:
```bash
python gcn_lstm_seq2seq.py   # Static GCN-LSTM baseline
python lstm_seq2seq.py       # Pure LSTM baseline
```

All outputs are saved to `results_dgcn/` (proposed model), `results_gcn/` (GCN baseline), and `results_lstm/` (LSTM baseline), respectively.

---

## 📁 Repository Structure

```
Flood-Prediction/
├── d_gcn_lstm_seq2seq.py    # Proposed D-GCN-LSTM-Seq2Seq
├── gcn_lstm_seq2seq.py      # Static GCN-LSTM baseline
├── lstm_seq2seq.py          # Pure LSTM baseline
├── dataset/                 # CSV files (7 physical fields, 24 probes)
├── results_dgcn/            # Outputs: plots, CSV metrics, model weights
├── results_gcn/             # Baseline outputs
├── results_lstm/            # Baseline outputs
├── requirements.txt
└── LICENSE
```

**Dataset**: `dataset/` contains `probe_coords.csv` and 7 feature files (`Ux`, `Uy`, `Uz`, `k`, `p`, `nut`, `ε`) at 24 virtual probe locations, downsampled to 5‑minute intervals. Raw simulation: 22,800 s → 76 time steps → 33 training sequences via sliding window (`INPUT_WINDOW=32`, `OUTPUT_WINDOW=12`). Features are standardized using `StandardScaler` fitted on the training set only.

---

## 📈 Performance Results

| Model | MSE | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- | :--- |
| **D-GCN-LSTM-Seq2Seq** (Proposed) | **0.000582** | **0.0241** | **0.0182** | **0.999** |
| GCN-LSTM (Baseline) | 0.00270 | 0.0519 | 0.0303 | 0.997 |
| Pure LSTM (Baseline) | 0.00392 | 0.0626 | 0.0368 | 0.996 |

**D-GCN reduces MSE by 85.2% vs. LSTM and 78.4% vs. GCN-LSTM.**

---

## 🧠 Model Architecture (Proposed)

```
Velocity fields (7 channels)
        ↓
VGDAG (velocity-guided dynamic graph)
        ↓
Sparse Edge Attention
        ↓
2-layer GCN
        ↓
LSTM Encoder (32 steps)
        ↓
Bahdanau Attention
        ↓
LSTM Decoder (12 steps)
        ↓
Physical Constraint Loss
        ↓
Flood front centroid extraction
```

See code docstrings for detailed mathematical formulations.

---

## 📊 Output Files

| Category | Files |
| :--- | :--- |
| **Loss curves** | `training_loss_curve.png`, `train_val_metrics_curve.png` |
| **Metrics** | `metrics_per_timestep_curve.png`, `test_metrics_bar.png`, `test_evaluation_metrics.csv`, `test_metrics_per_timestep.csv` |
| **Scatter plots** | `true_vs_pred_*.png` (All Features + 7 individual features) |
| **Front trajectory** | `true_pred_front_trajectory_combined.png`, `pred_flood_front_trajectory.png`, `front_trajectory_velocity_labels.png`, `front_instant_velocity.png`, `front_velocity_distribution.png`, `velocity_true_vs_pred.png` |
| **Graph visualization** | `spatial_graph_with_weights.png` (D-GCN only) · `graph_structure.png` (GCN only) |

---

## 💻 Requirements

- Python 3.10.11
- Dependencies listed in `requirements.txt`

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{yu2026flood,
  title     = {An Efficient Flood Prediction Method Considering Upstream-Downstream Correlations with Limited Data},
  author    = {Yu, Qingran},
  journal   = {HAUSCR YSA},
  year      = {2026}
}
```

---

## 📧 Contact

For questions or issues, please open a GitHub Issue.

---

**Last updated**: July 2026
