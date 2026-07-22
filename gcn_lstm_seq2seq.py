import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import os
import networkx as nx
import random
from typing import Tuple, Dict, List, Optional, Any

# ===================== Smart Backend =====================
if not os.environ.get('DISPLAY', ''):
    matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# ===================== Global Configuration =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "results_gcn")
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.unicode_minus': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300
})

def set_random_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    """Hyperparameter configuration class. All parameters are centrally managed."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    COORD_PATH = os.path.join(DATASET_DIR, "probe_coords.csv")
    FEAT_PATHS = {
        "Ux": os.path.join(DATASET_DIR, "Ux.csv"),
        "Uy": os.path.join(DATASET_DIR, "Uy.csv"),
        "Uz": os.path.join(DATASET_DIR, "Uz.csv"),
        "k": os.path.join(DATASET_DIR, "k.csv"),
        "p": os.path.join(DATASET_DIR, "p.csv"),
        "nut": os.path.join(DATASET_DIR, "nut.csv"),
        "epsilon": os.path.join(DATASET_DIR, "epsilon.csv")
    }
    TIME_COL_NAME = "Time (s)"
    DOWNSAMPLE_INTERVAL = 300
    INPUT_WINDOW = 32
    SLIDE_STEP = 1
    OUTPUT_WINDOW = 12
    NODE_FEAT_DIM = 7
    HIDDEN_DIM = 128
    ATTENTION_DIM = 128
    BATCH_SIZE = 1
    EPOCHS = 40
    LR = 5e-5
    TEACHER_FORCING_RATIO = 0.7
    KNN_K = 4
    TARGET_PROBES = 24
    NUM_FRONT_NODES = 8
    COLOR_PALETTE = {"train_loss": "#2E86AB", "val_loss": "#A23B72", "front": "#6A994E"}

# ===================== Evaluation Metrics =====================
def calculate_evaluation_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """Calculate overall evaluation metrics: MSE, RMSE, MAE, R²."""
    y_true_np = y_true.detach().cpu().numpy().reshape(-1)
    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    mse = np.mean((y_true_np - y_pred_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    r2 = r2_score(y_true_np, y_pred_np)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

def calculate_metrics_per_timestep(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[List[Dict], pd.DataFrame]:
    """Compute evaluation metrics per time step, returning a list and a DataFrame."""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    output_window = y_true_np.shape[1]
    metrics_list = []
    for t in range(output_window):
        true_t = y_true_np[:, t, :, :].reshape(-1)
        pred_t = y_pred_np[:, t, :, :].reshape(-1)
        mse = np.mean((true_t - pred_t) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_t - pred_t))
        try:
            r2 = r2_score(true_t, pred_t)
        except:
            r2 = np.nan
        metrics_list.append({
            "time_step": t + 1,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2
        })
    return metrics_list, pd.DataFrame(metrics_list)

# ===================== Physical Constraint Loss =====================
def physical_constraint_loss(preds_scaled: torch.Tensor, probe_coords: np.ndarray, scaler: StandardScaler) -> torch.Tensor:
    """
    Physical constraint loss: velocity bounds, spatial gradient, flow direction, and flow smoothness.
    """
    batch_size, output_window, num_probes, _ = preds_scaled.shape
    device = preds_scaled.device
    constraint_loss = 0.0

    preds_flat = preds_scaled.reshape(-1, Config.NODE_FEAT_DIM).detach().cpu().numpy()
    preds_denorm = scaler.inverse_transform(preds_flat)
    preds = torch.tensor(preds_denorm, dtype=torch.float32, device=device).reshape(preds_scaled.shape)

    pred_vel_mag = torch.sqrt(preds[..., 0]**2 + preds[..., 1]**2 + preds[..., 2]**2)
    upper_bound = 1.6
    lower_bound = 0.0
    velocity_upper_loss = torch.mean(torch.relu(pred_vel_mag - upper_bound))
    velocity_lower_loss = torch.mean(torch.relu(lower_bound - pred_vel_mag))
    constraint_loss += 0.5 * (velocity_upper_loss + velocity_lower_loss)

    kdtree = KDTree(probe_coords)
    _, neighbors = kdtree.query(probe_coords, k=3)
    neighbor_pairs = [(i, j) for i in range(num_probes) for j in neighbors[i][1:]]
    spatial_grad_loss = 0.0
    for (i, j) in neighbor_pairs:
        dx = probe_coords[i, 0] - probe_coords[j, 0]
        dy = probe_coords[i, 1] - probe_coords[j, 1]
        dz = probe_coords[i, 2] - probe_coords[j, 2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
        vel_i = torch.sqrt(preds[..., i, 0]**2 + preds[..., i, 1]**2 + preds[..., i, 2]**2)
        vel_j = torch.sqrt(preds[..., j, 0]**2 + preds[..., j, 1]**2 + preds[..., j, 2]**2)
        vel_diff = torch.abs(vel_i - vel_j)
        spatial_grad_loss += torch.mean(torch.relu(vel_diff / distance - 0.1))
    constraint_loss += 0.3 * (spatial_grad_loss / len(neighbor_pairs))

    pred_uy = preds[..., 1]
    pred_vel_mag_safe = pred_vel_mag + 1e-6
    cos_angle = pred_uy / pred_vel_mag_safe
    direction_loss = torch.mean(torch.relu(0.5 - cos_angle))
    constraint_loss += 0.2 * direction_loss

    total_vel_per_slice = torch.sum(pred_vel_mag, dim=2)
    if output_window > 1:
        flow_diff = torch.abs(total_vel_per_slice[:, 1:] - total_vel_per_slice[:, :-1])
        flow_loss = torch.mean(flow_diff)
        constraint_loss += 0.1 * flow_loss

    return constraint_loss

# ===================== Build Static KNN Graph =====================
def build_static_knn_graph(probe_coords: np.ndarray) -> torch.Tensor:
    """
    Construct a static undirected KNN graph based on Euclidean distance.

    Args:
        probe_coords: (N, 3) probe coordinates.

    Returns:
        edge_index: (2, E) edge indices.
    """
    print("\n" + "="*20 + " Construct Standard Static KNN Graph " + "="*20)
    N = len(probe_coords)
    kdtree = KDTree(probe_coords)
    edge_list = []

    for src in range(N):
        _, indices = kdtree.query(probe_coords[src], k=Config.KNN_K+1)
        for dst in indices[1:]:
            edge_list.append([src, dst])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(Config.DEVICE)
    return edge_index

# ===================== Static Graph Visualization =====================
def plot_static_graph(probe_coords: np.ndarray, edge_index: torch.Tensor) -> None:
    """
    Visualize the static KNN graph without a title, only axis labels.
    This matches the style of the main DGCN code, which does not use a title for graph plots.

    Args:
        probe_coords: (N, 3) probe coordinates.
        edge_index: (2, E) edge indices.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    G = nx.Graph()
    for i in range(len(probe_coords)):
        G.add_node(i, pos=(probe_coords[i, 0], probe_coords[i, 1]))
    edges = edge_index.cpu().numpy().T
    G.add_edges_from(edges)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200, node_color="#a1caf1", edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#444444", alpha=0.6, width=0.8)
    nx.draw_networkx_labels(G, pos, labels={i: str(i+1) for i in G.nodes()}, font_size=8)
    ax.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontweight='bold')
    # No title – intentionally left blank to match DGCN style
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "graph_structure.png"), bbox_inches='tight')
    plt.close()
    print("✅ Static graph saved to graph_structure.png")

# ===================== Flood Front Extraction & Velocity =====================
def calculate_front_velocity(front_coords: np.ndarray, time_interval: float = 5.0) -> Dict[str, Any]:
    """Compute instantaneous and average velocity of the flood front."""
    print("\n" + "="*20 + " Calculate Flood Front Velocity " + "="*20)
    num_timesteps = len(front_coords)
    if num_timesteps < 2:
        print("⚠️ Insufficient front coordinates for velocity calculation.")
        return {
            "instant_velocity": [], "avg_velocity": 0.0,
            "velocity_dir_x": [], "velocity_dir_y": [],
            "time_min": [], "front_coords_x": [], "front_coords_y": []
        }

    instant_velocity = np.zeros(num_timesteps)
    velocity_dir_x = np.zeros(num_timesteps)
    velocity_dir_y = np.zeros(num_timesteps)
    instant_velocity[0] = 0.0

    for t in range(1, num_timesteps):
        dx = front_coords[t, 0] - front_coords[t-1, 0]
        dy = front_coords[t, 1] - front_coords[t-1, 1]
        dz = front_coords[t, 2] - front_coords[t-1, 2]
        displacement = np.sqrt(dx**2 + dy**2 + dz**2)
        instant_velocity[t] = displacement / time_interval
        if displacement > 1e-6:
            velocity_dir_x[t] = dx / displacement
            velocity_dir_y[t] = dy / displacement

    total_displacement = np.sqrt(
        (front_coords[-1,0]-front_coords[0,0])**2 +
        (front_coords[-1,1]-front_coords[0,1])**2 +
        (front_coords[-1,2]-front_coords[0,2])**2
    )
    total_time = (num_timesteps - 1) * time_interval
    avg_velocity = total_displacement / total_time if total_time > 0 else 0.0

    for t in range(num_timesteps):
        print(f"Time {t*5}min: Instantaneous velocity = {instant_velocity[t]:.4f} m/min | Direction = ({velocity_dir_x[t]:.4f}, {velocity_dir_y[t]:.4f})")

    velocity_data = {
        "time_min": [t*time_interval for t in range(num_timesteps)],
        "instant_velocity": instant_velocity,
        "avg_velocity": avg_velocity,
        "velocity_dir_x": velocity_dir_x,
        "velocity_dir_y": velocity_dir_y,
        "front_coords_x": front_coords[:, 0],
        "front_coords_y": front_coords[:, 1]
    }
    return velocity_data

def extract_flood_front(predictions: torch.Tensor, scaler: StandardScaler,
                        probe_coords: np.ndarray) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Extract flood front centroid from predicted velocity fields.

    Why we don't take the minimum-Y nodes directly:
    - The very tip of the front has only 1–2 sparse nodes.
    - Those nodes alternate due to numerical noise.
    - The centroid would jump erratically.

    Our approach:
    - Rank nodes by descending Y (downstream to upstream).
    - Start with the 8 most downstream nodes.
    - Add one upstream node at each time step.
    - Compute the centroid of the entire growing set.

    Effect:
    - Smooths out tip fluctuations (low-pass filtering).
    - Tracks the center of mass of the flooded region.
    - Gives a stable, continuous trajectory.
    """
    print("\n" + "="*20 + " Extract Flood Front " + "="*20)
    pred_flat = predictions.cpu().numpy().reshape(-1, Config.NODE_FEAT_DIM)
    pred_denorm = scaler.inverse_transform(pred_flat).reshape(predictions.shape)
    x_coords = probe_coords[:, 0]
    y_coords = probe_coords[:, 1]

    pred_ux_seq = pred_denorm[0, :, :, 0]
    pred_uy_seq = pred_denorm[0, :, :, 1]

    output_slices = pred_ux_seq.shape[0]
    base_front_size = Config.NUM_FRONT_NODES

    sorted_nodes = np.lexsort((x_coords, -y_coords)).tolist()
    current_front_ids = sorted_nodes[:base_front_size]
    front_coords = []

    for t in range(output_slices):
        actual_time = t * 5
        if len(current_front_ids) < len(sorted_nodes):
            current_front_ids.append(sorted_nodes[len(current_front_ids)])
        current_front_ids = list(dict.fromkeys(current_front_ids))

        t_ux = pred_ux_seq[t, current_front_ids]
        t_uy = pred_uy_seq[t, current_front_ids]
        valid_mask = t_uy > -0.05
        valid_ids = np.array(current_front_ids)[valid_mask]

        if len(valid_ids) < base_front_size:
            fallback_nodes = sorted(current_front_ids, key=lambda idx: probe_coords[idx, 1])
            valid_ids = np.array(fallback_nodes[:base_front_size])

        centroid = np.mean(probe_coords[valid_ids], axis=0)
        front_coords.append(centroid)
        print(f"Time {actual_time}min: Front centroid ({centroid[0]:.2f}, {centroid[1]:.2f})")

    front_coords = np.array(front_coords)
    velocity_data = calculate_front_velocity(front_coords)
    return front_coords, velocity_data, {"avg_velocity": velocity_data["avg_velocity"]}

def extract_true_flood_front(true_data: torch.Tensor, scaler: StandardScaler,
                             probe_coords: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Extract flood front trajectory from ground truth (same logic as prediction)."""
    print("\n" + "="*20 + " Extract True Flood Front" + "="*20)
    true_flat = true_data.cpu().numpy().reshape(-1, Config.NODE_FEAT_DIM)
    true_denorm = scaler.inverse_transform(true_flat).reshape(true_data.shape) 
    x_coords = probe_coords[:, 0]
    y_coords = probe_coords[:, 1]

    pred_ux_seq = true_denorm[0, :, :, 0]
    pred_uy_seq = true_denorm[0, :, :, 1]

    output_slices = pred_ux_seq.shape[0]
    base_front_size = Config.NUM_FRONT_NODES

    sorted_nodes = np.lexsort((x_coords, -y_coords)).tolist()
    current_front_ids = sorted_nodes[:base_front_size]
    front_coords = []

    for t in range(output_slices):
        actual_time = t * 5
        if len(current_front_ids) < len(sorted_nodes):
            current_front_ids.append(sorted_nodes[len(current_front_ids)])
        current_front_ids = list(dict.fromkeys(current_front_ids))

        t_ux = pred_ux_seq[t, current_front_ids]
        t_uy = pred_uy_seq[t, current_front_ids]
        valid_mask = t_uy > -0.05
        valid_ids = np.array(current_front_ids)[valid_mask]

        if len(valid_ids) < base_front_size:
            fallback_nodes = sorted(current_front_ids, key=lambda idx: probe_coords[idx, 1])
            valid_ids = np.array(fallback_nodes[:base_front_size])

        centroid = np.mean(probe_coords[valid_ids], axis=0)
        front_coords.append(centroid)
        print(f"True Time {actual_time}min: Front centroid ({centroid[0]:.2f}, {centroid[1]:.2f})")

    front_coords = np.array(front_coords)
    velocity_data = calculate_front_velocity(front_coords)
    return front_coords, velocity_data

# ===================== Visualization Functions =====================
def plot_training_loss(train_losses: List[float], val_losses: List[float]) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', color=Config.COLOR_PALETTE["train_loss"], linewidth=2)
    ax.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', color=Config.COLOR_PALETTE["val_loss"], linewidth=2)
    best_val_idx = np.argmin(val_losses)
    ax.scatter(best_val_idx+1, val_losses[best_val_idx], color='red', s=60, label=f'Best Val Loss: {val_losses[best_val_idx]:.6f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training vs Validation Loss Curve', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "training_loss_curve.png"), bbox_inches='tight')
    plt.close()

def plot_train_val_metrics(train_metrics_hist: List[Dict], val_metrics_hist: List[Dict]) -> None:
    """Plot training and validation metrics (MSE, RMSE, MAE, R²) across epochs."""
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        train_vals = [epoch_metrics[metric] for epoch_metrics in train_metrics_hist]
        val_vals = [epoch_metrics[metric] for epoch_metrics in val_metrics_hist]
        ax.plot(range(1, Config.EPOCHS+1), train_vals, label=f'Train {metric}', color=colors[idx], linewidth=2, marker='o', markersize=3)
        ax.plot(range(1, Config.EPOCHS+1), val_vals, label=f'Val {metric}', color=colors[idx], linestyle='--', linewidth=2, marker='s', markersize=3)
        ax.set_title(f'{metric} Curve', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(alpha=0.3)
        if metric == 'R²':
            ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "train_val_metrics_curve.png"), bbox_inches='tight')
    plt.close()

def plot_timestep_metrics_curve(metrics_df: pd.DataFrame) -> None:
    """
    Plot per‑timestep MSE, RMSE, and MAE curves.
    MAE is placed in the centre of the second row.
    """
    timesteps = metrics_df["time_step"].values
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
    ax1.plot(timesteps, metrics_df["MSE"], color=colors[0], linewidth=3, marker='o', markersize=8)
    for x, y in zip(timesteps, metrics_df["MSE"]):
        ax1.text(x, y, f'{y:.4f}', fontsize=8, ha='center', va='bottom')
    ax1.set_title('MSE Across 12 Prediction Steps', fontweight='bold')
    ax1.set_xlabel('Prediction Time Step', fontweight='bold')
    ax1.set_ylabel('MSE', fontweight='bold')
    ax1.set_xticks(timesteps)
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
    ax2.plot(timesteps, metrics_df["RMSE"], color=colors[1], linewidth=3, marker='o', markersize=8)
    for x, y in zip(timesteps, metrics_df["RMSE"]):
        ax2.text(x, y, f'{y:.4f}', fontsize=8, ha='center', va='bottom')
    ax2.set_title('RMSE Across 12 Prediction Steps', fontweight='bold')
    ax2.set_xlabel('Prediction Time Step', fontweight='bold')
    ax2.set_ylabel('RMSE', fontweight='bold')
    ax2.set_xticks(timesteps)
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot2grid((2, 4), (1, 1), colspan=2)
    ax3.plot(timesteps, metrics_df["MAE"], color=colors[2], linewidth=3, marker='o', markersize=8)
    for x, y in zip(timesteps, metrics_df["MAE"]):
        ax3.text(x, y, f'{y:.4f}', fontsize=8, ha='center', va='bottom')
    ax3.set_title('MAE Across 12 Prediction Steps', fontweight='bold')
    ax3.set_xlabel('Prediction Time Step', fontweight='bold')
    ax3.set_ylabel('MAE', fontweight='bold')
    ax3.set_xticks(timesteps)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "metrics_per_timestep_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Metrics per timestep curve saved to metrics_per_timestep_curve.png")

def plot_test_metrics_bar(test_metrics: Dict[str, float]) -> None:
    """Plot a bar chart of test set evaluation metrics."""
    metric_names = list(test_metrics.keys())
    metric_values = list(test_metrics.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(metric_values)*0.02),
                f'{val:.6f}', ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel('Evaluation Metric')
    ax.set_ylabel('Value')
    ax.set_title('Test Set Evaluation Metrics', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "test_metrics_bar.png"), bbox_inches='tight')
    plt.close()

def plot_true_vs_pred(y_true: torch.Tensor, y_pred: torch.Tensor, feature_name: str = "All Features") -> None:
    """Scatter plot of ground truth vs predictions with R² annotation."""
    y_true_np = y_true.cpu().numpy().reshape(-1)
    y_pred_np = y_pred.cpu().numpy().reshape(-1)
    r2 = r2_score(y_true_np, y_pred_np)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_np, y_pred_np, alpha=0.6, s=30, color='purple', edgecolors='black', linewidth=0.2)
    min_val = min(y_true_np.min(), y_pred_np.min())
    max_val = max(y_true_np.max(), y_pred_np.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction Line (y=x)')
    ax.text(0.05, 0.95, f'R² = {r2:.6f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12, fontweight='bold')
    ax.set_xlabel('Ground Truth (Standardized)')
    ax.set_ylabel('Prediction (Standardized)')
    ax.set_title(f'Ground Truth vs Prediction | {feature_name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'true_vs_pred_{feature_name.replace(" ", "_")}.png'), bbox_inches='tight')
    plt.close()

def plot_flood_front_trajectory_with_velocity(front_coords: np.ndarray, velocity_data: Dict,
                                              probe_coords: np.ndarray) -> None:
    """Generate three independent figures: trajectory with velocity labels, velocity curve, and velocity distribution."""
    time_min = velocity_data["time_min"]
    instant_velocity = velocity_data["instant_velocity"]

    # Trajectory with velocity labels
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.scatter(probe_coords[:, 0], probe_coords[:, 1], color='gray', alpha=0.5, label='Probe Locations')
    ax1.plot(front_coords[:, 0], front_coords[:, 1],
             color=Config.COLOR_PALETTE["front"], linewidth=3, marker='o', label='Flood Front Trajectory')
    for i, (x, y) in enumerate(front_coords[:, :2]):
        ax1.annotate(f'{instant_velocity[i]:.2f} m/min', (x, y),
                     xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.legend()
    ax1.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "front_trajectory_velocity_labels.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Instantaneous velocity over time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(time_min, instant_velocity, color='#FF6B6B', linewidth=3, marker='o')
    ax2.axhline(y=velocity_data["avg_velocity"], color='red', linestyle='--',
                label=f'Avg Velocity: {velocity_data["avg_velocity"]:.4f} m/min')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Instant Velocity (m/min)')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "front_instant_velocity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # Velocity distribution
    valid_velocity = instant_velocity[instant_velocity > 0]
    if len(valid_velocity) > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.hist(valid_velocity, bins=10, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Velocity (m/min)')
        ax3.set_ylabel('Frequency')
        ax3.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "front_velocity_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close(fig3)

    print("✅ Flood front trajectory and velocity figures saved.")

def plot_velocity_true_vs_pred(true_vel: Dict, pred_vel: Dict) -> None:
    """Plot true vs predicted velocity comparison with average velocity lines."""
    plt.figure(figsize=(10,5))
    t = true_vel['time_min']
    plt.plot(t, true_vel['instant_velocity'], color=Config.COLOR_PALETTE["train_loss"],
             linewidth=3, marker='o', label='True Velocity')
    plt.plot(t, pred_vel['instant_velocity'], color=Config.COLOR_PALETTE["val_loss"],
             linewidth=3, marker='s', linestyle='--', label='Pred Velocity')
    true_avg = true_vel['avg_velocity']
    pred_avg = pred_vel['avg_velocity']
    plt.axhline(y=true_avg, color=Config.COLOR_PALETTE["train_loss"], linestyle=':',
                linewidth=2, label=f'True Avg Velocity: {true_avg:.4f} m/min')
    plt.axhline(y=pred_avg, color=Config.COLOR_PALETTE["val_loss"], linestyle=':',
                linewidth=2, label=f'Pred Avg Velocity: {pred_avg:.4f} m/min')
    plt.xlabel('Time (min)')
    plt.ylabel('Velocity (m/min)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "velocity_true_vs_pred.png"))
    plt.close()

def plot_true_pred_front_trajectory(true_front: np.ndarray, pred_front: np.ndarray,
                                    probe_coords: np.ndarray) -> None:
    """Create a side‑by‑side comparison of true and predicted front trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.scatter(probe_coords[:, 0], probe_coords[:, 1], color='gray', alpha=0.5, label='Probe Locations')
    ax1.plot(pred_front[:, 0], pred_front[:, 1], color='#A23B72', linewidth=3, marker='s', label='Pred Front')
    for i, (x, y) in enumerate(pred_front[:, :2]):
        ax1.annotate(f'{i*5}min', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')
    ax1.set_title('Pred Flood Front Trajectory (Top View)', fontweight='bold')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.legend()
    ax1.axis('equal')

    ax2.scatter(probe_coords[:, 0], probe_coords[:, 1], color='gray', alpha=0.5, label='Probe Locations')
    ax2.plot(true_front[:, 0], true_front[:, 1], color='#2E86AB', linewidth=3, marker='o', label='True Front')
    for i, (x, y) in enumerate(true_front[:, :2]):
        ax2.annotate(f'{i*5}min', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')
    ax2.set_title('True Flood Front Trajectory (Top View)', fontweight='bold')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')
    ax2.legend()
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "true_pred_front_trajectory_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pred_flood_front(front_coords: np.ndarray, probe_coords: np.ndarray) -> None:
    """Plot the predicted flood front trajectory alone."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(probe_coords[:, 0], probe_coords[:, 1], color='gray', alpha=0.5, label='Probe Locations')
    ax.plot(front_coords[:, 0], front_coords[:, 1], color='#6A994E',
            linewidth=3, marker='o', label='Predicted Flood Front Trajectory')
    for i, (x, y) in enumerate(front_coords[:, :2]):
        ax.annotate(f'{i*5}min', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "pred_flood_front_trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ===================== Data Loading and Preprocessing =====================
def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, None, List[int], int]:
    """
    Load all CSV files, merge, downsample, and return raw feature tensor (no standardization yet).
    """
    if not os.path.exists(Config.COORD_PATH):
        raise FileNotFoundError(f"Coordinate file not found: {Config.COORD_PATH}. Please ensure dataset is placed in '{DATASET_DIR}'.")

    print("="*20 + " Load Probe Coordinates " + "="*20)
    coord_raw = pd.read_csv(Config.COORD_PATH, header=None, nrows=4)
    probe_ids = coord_raw.iloc[0, :].values
    x_coords = coord_raw.iloc[1, :].values
    y_coords = coord_raw.iloc[2, :].values
    z_coords = coord_raw.iloc[3, :].values
    coord_df = pd.DataFrame({
        "probe_id": probe_ids, "x": x_coords, "y": y_coords, "z": z_coords
    })
    coord_df = coord_df[coord_df["probe_id"].between(1, Config.TARGET_PROBES)].reset_index(drop=True)
    coord_df["probe_id"] = coord_df["probe_id"].astype(int)
    unique_probes = sorted(coord_df["probe_id"].unique())
    num_probes = len(unique_probes)
    print(f"Valid probe count: {num_probes}")

    print("\n" + "="*20 + " Load Feature Files " + "="*20)
    merged_feat_df = None

    for feat_name, csv_path in Config.FEAT_PATHS.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Feature file not found: {csv_path}")
        print(f"Processing {feat_name}.csv...")
        df = pd.read_csv(csv_path, sep=",", na_filter=False, low_memory=False, encoding="utf-8")
        if Config.TIME_COL_NAME not in df.columns:
            raise ValueError(f"❌ {csv_path} missing time column: {Config.TIME_COL_NAME}")

        df[Config.TIME_COL_NAME] = df[Config.TIME_COL_NAME].astype(np.float32).round().astype(np.int32)
        df = df[(df[Config.TIME_COL_NAME] >= 0) & (df[Config.TIME_COL_NAME] <= 22800)]

        probe_cols = [col for col in df.columns if col != Config.TIME_COL_NAME and col.strip()]
        if len(probe_cols) == 0:
            raise ValueError(f"❌ {csv_path} has no probe columns")

        df_long = df.melt(id_vars=[Config.TIME_COL_NAME], var_name="probe_id_str", value_name=feat_name)
        df_long.rename(columns={Config.TIME_COL_NAME: "time_step"}, inplace=True)
        df_long["probe_id_str"] = df_long["probe_id_str"].astype(str)
        df_long["probe_id"] = df_long["probe_id_str"].str.replace("", "").astype(int)
        df_long = df_long[df_long["probe_id"].isin(unique_probes)][["time_step", "probe_id", feat_name]]

        if merged_feat_df is None:
            merged_feat_df = df_long
        else:
            merged_feat_df = pd.merge(merged_feat_df, df_long, on=["time_step", "probe_id"], how="inner")

    unique_raw_timesteps = sorted(merged_feat_df["time_step"].unique())
    sampled_timesteps = [t for t in unique_raw_timesteps if t % Config.DOWNSAMPLE_INTERVAL == 0]
    if 22800 not in sampled_timesteps:
        sampled_timesteps.append(22800)
    sampled_timesteps = sorted(list(set(sampled_timesteps)))
    sampled_df = merged_feat_df[merged_feat_df["time_step"].isin(sampled_timesteps)]
    num_sampled_timesteps = len(sampled_timesteps)

    min_required = Config.INPUT_WINDOW + Config.OUTPUT_WINDOW
    if num_sampled_timesteps < min_required:
        raise ValueError(f"❌ Insufficient slices! Required {min_required}, current {num_sampled_timesteps}")
    print(f"Downsampling completed: Original {len(unique_raw_timesteps)} steps → {num_sampled_timesteps} steps")

    feat_names = list(Config.FEAT_PATHS.keys())
    feat_tensor = np.zeros((num_sampled_timesteps, num_probes, Config.NODE_FEAT_DIM))
    probe_coords = np.zeros((num_probes, 3))
    for i, pid in enumerate(unique_probes):
        probe_coords[i] = coord_df[coord_df["probe_id"] == pid][["x", "y", "z"]].values[0]

    for t_idx, raw_t in enumerate(sampled_timesteps):
        t_data = sampled_df[sampled_df["time_step"] == raw_t]
        for p_idx, pid in enumerate(unique_probes):
            p_data = t_data[t_data["probe_id"] == pid]
            if not p_data.empty:
                feat_tensor[t_idx, p_idx] = p_data[feat_names].values[0]
                
    scaler = None
    print(f"Data loading completed: Feature tensor shape {feat_tensor.shape} (raw, unscaled)")
    return feat_tensor, probe_coords, scaler, unique_probes, num_probes

# ===================== Sliding Window Samples & Dataset =====================
def create_sliding_window_samples(feat_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding‑window samples and split into train/validation/test sets."""
    num_slices, num_probes, _ = feat_tensor.shape
    min_required = Config.INPUT_WINDOW + Config.OUTPUT_WINDOW
    if num_slices < min_required:
        raise ValueError(f"❌ Insufficient slices! Required {min_required}, current {num_slices}")

    samples = []
    total_available = num_slices - min_required + 1
    for start in range(0, total_available, Config.SLIDE_STEP):
        history = feat_tensor[start:start+Config.INPUT_WINDOW]
        future = feat_tensor[start+Config.INPUT_WINDOW:start+min_required]
        samples.append((history, future))

    X = np.array([s[0] for s in samples])
    Y = np.array([s[1] for s in samples])
    total_samples = len(X)
    print(f"Sliding window samples generated: {total_samples}")

    if total_samples <= 3:
        train_X, train_Y = X, Y
        val_X, val_Y = X[:1], Y[:1]
        test_X, test_Y = X[:1], Y[:1]
    else:
        train_size = max(int(0.7 * total_samples), 1)
        val_size = max(int(0.15 * total_samples), 1)
        test_size = total_samples - train_size - val_size
        train_X, train_Y = X[:train_size], Y[:train_size]
        val_X, val_Y = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
        test_X, test_Y = X[-test_size:], Y[-test_size:]

    print(f"Dataset split: Train {len(train_X)} | Val {len(val_X)} | Test {len(test_X)}")
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

class FloodDataset(Dataset):
    """Custom dataset that converts numpy arrays to torch tensors and moves to device."""
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).to(Config.DEVICE)
        self.Y = torch.tensor(Y, dtype=torch.float32).to(Config.DEVICE)
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

# ===================== Model Definitions =====================
class StandardGCN(nn.Module):
    """Standard two-layer GCN with residual connection and layer norm."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x.reshape(-1, C)
        edge_index = edge_index.repeat(1, B)
        x = self.relu(self.gcn1(x, edge_index))
        x = self.norm(x)
        x = self.dropout(x)
        x = self.relu(self.gcn2(x, edge_index)) + x
        return x.reshape(B, N, -1)

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism."""
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.w_en = nn.Linear(hidden_dim, attn_dim)
        self.w_de = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, enc_out: torch.Tensor, dec_hid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dec_hid_last = dec_hid[-1]
        enc_proj = self.w_en(enc_out)
        dec_proj = self.w_de(dec_hid_last).unsqueeze(1)
        score = self.v(torch.tanh(enc_proj + dec_proj))
        attn_w = self.softmax(score)
        ctx = torch.sum(enc_out * attn_w, dim=1)
        return ctx, attn_w

class EncoderLSTM(nn.Module):
    """LSTM encoder that flattens nodes and time."""
    def __init__(self, in_dim: int, hid_dim: int, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, n_layers, batch_first=True,
                            dropout=0.1 if n_layers>1 else 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, N, C = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B*N, T, C)
        enc_out, (h_n, c_n) = self.lstm(x_reshaped)
        return enc_out, (h_n, c_n)

class DecoderLSTM(nn.Module):
    """LSTM decoder with attention."""
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, attn_dim: int, n_layers: int = 2):
        super().__init__()
        self.attn = BahdanauAttention(hid_dim, attn_dim)
        self.lstm = nn.LSTM(in_dim + hid_dim, hid_dim, n_layers, batch_first=True,
                            dropout=0.1 if n_layers>1 else 0)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x_t: torch.Tensor, enc_out: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, _ = x_t.shape
        x_reshaped = x_t.reshape(B*N, -1)
        ctx, _ = self.attn(enc_out, h)
        lstm_input = torch.cat([x_reshaped.unsqueeze(1), ctx.unsqueeze(1)], dim=-1)
        out, (h_new, c_new) = self.lstm(lstm_input, (h, c))
        output = self.fc(out.squeeze(1)).reshape(B, N, -1)
        return output, (h_new, c_new)

class GCNLSTM(nn.Module):
    """Static GCN-LSTM model with attention decoder."""
    def __init__(self, node_feat_dim: int, hidden_dim: int, out_dim: int, out_seq: int, attn_dim: int):
        super().__init__()
        self.gcn = StandardGCN(node_feat_dim, hidden_dim)
        self.enc = EncoderLSTM(hidden_dim, hidden_dim)
        self.dec = DecoderLSTM(out_dim, hidden_dim, out_dim, attn_dim)
        self.init_input = nn.Linear(hidden_dim, out_dim)
        self.out_seq = out_seq

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T_in, N, _ = x.shape
        gcn_feat_list = []
        for t in range(T_in):
            gcn_out = self.gcn(x[:, t], edge_index)
            gcn_feat_list.append(gcn_out)
        gcn_feat = torch.stack(gcn_feat_list, dim=1)
        enc_out, (h, c) = self.enc(gcn_feat)
        x_t = self.init_input(gcn_feat[:, -1])
        outputs = []
        for t in range(self.out_seq):
            x_t, (h, c) = self.dec(x_t, enc_out, h, c)
            outputs.append(x_t)
            if self.training and target is not None and torch.rand(1).item() < Config.TEACHER_FORCING_RATIO:
                x_t = target[:, t]
        return torch.stack(outputs, dim=1)

# ===================== Main Training Pipeline =====================
def train_and_predict() -> None:
    """Main training pipeline for static GCN-LSTM baseline."""
    set_random_seed(42)

    feat_tensor, probe_coords, _, _, num_probes = load_and_preprocess_data()

    edge_index = build_static_knn_graph(probe_coords)

    plot_static_graph(probe_coords, edge_index)

    train_X, train_Y, val_X, val_Y, test_X, test_Y = create_sliding_window_samples(feat_tensor)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, Config.NODE_FEAT_DIM))

    def standardize(data: np.ndarray) -> np.ndarray:
        orig_shape = data.shape
        flat = data.reshape(-1, Config.NODE_FEAT_DIM)
        scaled = scaler.transform(flat)
        return scaled.reshape(orig_shape)

    train_X = standardize(train_X)
    train_Y = standardize(train_Y)
    val_X   = standardize(val_X)
    val_Y   = standardize(val_Y)
    test_X  = standardize(test_X)
    test_Y  = standardize(test_Y)

    train_loader = DataLoader(FloodDataset(train_X, train_Y), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(FloodDataset(val_X, val_Y), batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(FloodDataset(test_X, test_Y), batch_size=Config.BATCH_SIZE, shuffle=False)

    model = GCNLSTM(
        node_feat_dim=Config.NODE_FEAT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        out_dim=Config.NODE_FEAT_DIM,
        out_seq=Config.OUTPUT_WINDOW,
        attn_dim=Config.ATTENTION_DIM
    ).to(Config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics_hist = []
    val_metrics_hist = []

    print("\n" + "="*20 + " Start Training Static GCN-LSTM " + "="*20)
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X, edge_index, batch_Y)
            mse_loss = criterion(preds, batch_Y)
            phys_loss = physical_constraint_loss(preds, probe_coords, scaler)
            total_loss = mse_loss + 0.5 * phys_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item() * batch_X.size(0)
            train_preds.append(preds)
            train_targets.append(batch_Y)

        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_metrics = calculate_evaluation_metrics(train_targets, train_preds)
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        train_metrics_hist.append(train_metrics)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                preds = model(batch_X, edge_index)
                mse_loss = criterion(preds, batch_Y)
                phys_loss = physical_constraint_loss(preds, probe_coords, scaler)
                total_loss = mse_loss + 0.5 * phys_loss
                val_loss += total_loss.item() * batch_X.size(0)
                val_preds.append(preds)
                val_targets.append(batch_Y)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_metrics = calculate_evaluation_metrics(val_targets, val_preds)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_metrics_hist.append(val_metrics)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"Train R²: {train_metrics['R²']:.6f} | Val R²: {val_metrics['R²']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_gcn_model.pth"))
            print(f"✅ Best model saved (Val Loss: {best_val_loss:.6f})")

    plot_training_loss(train_losses, val_losses)
    plot_train_val_metrics(train_metrics_hist, val_metrics_hist)

    print("\n" + "="*20 + " Test Phase (Best Model) " + "="*20)
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, "best_gcn_model.pth")))
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            preds = model(batch_X, edge_index)
            test_preds.append(preds)
            test_targets.append(batch_Y)
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)

    timestep_metrics_list, timestep_metrics_df = calculate_metrics_per_timestep(test_targets, test_preds)
    timestep_metrics_df.to_csv(os.path.join(RESULT_DIR, "test_metrics_per_timestep.csv"), index=False)
    print("\n=== Per-Timestep Evaluation Metrics ===")
    print(timestep_metrics_df.round(6))

    test_metrics = calculate_evaluation_metrics(test_targets, test_preds)
    test_phys_loss = physical_constraint_loss(test_preds, probe_coords, scaler).item()
    print("\nTest Set Overall Evaluation Results:")
    print(f"Physical Constraint Loss: {test_phys_loss:.6f}")
    for metric, value in test_metrics.items():
        print(f"{metric}：{value:.6f}")

    test_metrics_df = pd.DataFrame({
        "MSE": [test_metrics['MSE']],
        "RMSE": [test_metrics['RMSE']],
        "MAE": [test_metrics['MAE']],
        "R²": [test_metrics['R²']],
        "Physical_Constraint_Loss": [test_phys_loss]
    })
    test_metrics_df.to_csv(os.path.join(RESULT_DIR, "test_evaluation_metrics.csv"), index=False)

    plot_test_metrics_bar(test_metrics)
    plot_timestep_metrics_curve(timestep_metrics_df)
    plot_true_vs_pred(test_targets, test_preds, "All Features")
    feat_names = list(Config.FEAT_PATHS.keys())
    for feat_idx, feat_name in enumerate(feat_names):
        plot_true_vs_pred(test_targets[..., feat_idx], test_preds[..., feat_idx], feat_name)

    pred_front, pred_vel, _ = extract_flood_front(test_preds, scaler, probe_coords)
    true_front, true_vel = extract_true_flood_front(test_targets, scaler, probe_coords)

    plot_true_pred_front_trajectory(true_front, pred_front, probe_coords)
    plot_pred_flood_front(pred_front, probe_coords)
    plot_velocity_true_vs_pred(true_vel, pred_vel)
    plot_flood_front_trajectory_with_velocity(pred_front, pred_vel, probe_coords)

    print("\n✅ All tasks completed! Results saved to:", RESULT_DIR)

# ===================== Entry Point =====================
if __name__ == "__main__":
    try:
        train_and_predict()
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
