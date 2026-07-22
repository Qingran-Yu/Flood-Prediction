# Flood-Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> Official implementation for the paper accepted by **HAUSCR YSA 2026**. Released to ensure full reproducibility.

---

## 📦 Overview

This repository implements three models for flood front prediction using CFD probe data—the virtual counterpart to the **Mobile Spherical Sensing Unit (MSSU)** hardware:

| Model | File | Description |
| :--- | :--- | :--- |
| **D-GCN-LSTM-Seq2Seq (Proposed)** | `d_gcn_lstm_seq2seq.py` | Velocity-Guided Dynamic Acyclic Graph (VGDAG) with sparse edge attention + LSTM Seq2Seq. Graph topology adapts to real-time flow velocity and direction. |
| **GCN-LSTM (Baseline)** | `gcn_lstm_seq2seq.py` | Standard GCN with static KNN graph (K=4) + LSTM Seq2Seq. Graph topology is fixed. |
| **Pure LSTM (Baseline)** | `lstm_seq2seq.py` | No spatial graph. All node features flattened into a single vector. Pure temporal baseline. |

**Fair comparison guarantee**: All three models share identical optimization settings (`AdamW`, `lr=5e-5`, `weight_decay=1e-4`, gradient clipping `max_norm=1.0`) and random seed 42. Performance differences are attributable **solely to the graph structure** (dynamic vs. static vs. none).

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

> **Note**: If you are unsure, use Option A. It will work on any machine. For GPU acceleration, ensure you have NVIDIA drivers and CUDA 11.8 installed.

### 4. Run training + evaluation

**Proposed model**:
```bash
python d_gcn_lstm_seq2seq.py
```

**Baselines (for reproducing comparative experiments)**:
```bash
python gcn_lstm_seq2seq.py   # Static GCN-LSTM
python lstm_seq2seq.py       # Pure LSTM
```

All outputs are saved to `results_dgcn/`, `results_gcn/`, and `results_lstm/`, respectively.

---

## 📂 Dataset

All simulation data are in `dataset/`:

| File | Description |
|------|-------------|
| `probe_coords.csv` | 24 probe coordinates (x, y, z in meters) |
| `Ux.csv`, `Uy.csv`, `Uz.csv` | Velocity components (m/s) |
| `p.csv` | Pressure (Pa) |
| `k.csv`, `nut.csv`, `epsilon.csv` | Turbulence quantities |

**Simulation details**:
- Raw simulation: 22,800 s at 1 s intervals
- Downsampled to 300 s intervals → 76 time steps
- Sliding window: `INPUT_WINDOW=32`, `OUTPUT_WINDOW=12` → 33 training sequences

**Preprocessing**: `StandardScaler` fitted exclusively on the training set; same scaling applied to validation and test sets to prevent information leakage.

---

## 📊 Output Files

All outputs are saved to separate directories. Each script generates the following files in its respective `results_*` folder:

### Common Files (All Models)

| File | Description |
|------|-------------|
| `training_loss_curve.png` | Training and validation loss over 40 epochs |
| `train_val_metrics_curve.png` | MSE, RMSE, MAE, R² curves over epochs |
| `metrics_per_timestep_curve.png` | MSE, RMSE, MAE across 12 prediction steps (5–60 min) |
| `test_metrics_bar.png` | Test set metrics bar chart |
| `test_evaluation_metrics.csv` | Overall test metrics (MSE, RMSE, MAE, R²) |
| `test_metrics_per_timestep.csv` | Per-step metrics (12 rows) |
| `true_vs_pred_*.png` | Scatter plots: ground truth vs prediction (All Features + 7 individual features) |
| `true_pred_front_trajectory_combined.png` | True vs predicted flood front trajectory (side-by-side) |
| `pred_flood_front_trajectory.png` | Predicted flood front trajectory only |
| `front_trajectory_velocity_labels.png` | Front trajectory with velocity labels at each time step |
| `front_instant_velocity.png` | Instantaneous front velocity over time |
| `front_velocity_distribution.png` | Histogram of front velocities |
| `velocity_true_vs_pred.png` | True vs predicted front velocity comparison |

### Model-Specific Files

| Model | Additional Files |
| :--- | :--- |
| **D-GCN (Proposed)** | `spatial_graph_with_weights.png` — dynamic spatial graph with edge weights (color + width encode attention strength) |
| **GCN-LSTM (Baseline)** | `graph_structure.png` — static KNN graph (unweighted, fixed topology) |
| **Pure LSTM** | (none) |

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

## 💻 Requirements

Tested on:
- **Python 3.10.11**
- Dependencies are listed in `requirements.txt`

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{yu2026flood,
  title     = {An Efficient Flood Prediction Method Considering Upstream-Downstream Correlations with Limited Data},
  author    = {Yu, Qingran},
  journal   = {Young Scholars Academic (HAUSCR YSA)},
  year      = {2026},
  note      = {Accepted, in production}
}
```

---

## 📧 Contact

For questions or issues, please open a GitHub Issue.

---

**Last updated**: July 2026
