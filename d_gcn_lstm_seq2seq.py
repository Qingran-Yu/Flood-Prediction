import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.pyplot as plt
import warnings
import os
import networkx as nx
import random
from typing import Tuple, Dict, List, Optional, Any

# ===================== Smart Backend =====================
if not os.environ.get('DISPLAY', ''):
    matplotlib.use('Agg')   # Automatically enabled for headless servers

warnings.filterwarnings('ignore')

# ===================== Global Configuration =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "results_dgcn")
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
    EDGE_ATTR_DIM = 3
    HIDDEN_DIM = 128
    ATTENTION_DIM = 128
    BATCH_SIZE = 1
    EPOCHS = 40
    LR = 5e-5
    TEACHER_FORCING_RATIO = 0.7
    KNN_K = 4
    TARGET_PROBES = 24
    SPEED_THRESHOLD = 0.005
    BASE_RADIUS = 32.0
    RADIUS_SCALE_FACTOR = 2.0
    SPARSE_ATTN_HEADS = 2
    SPARSE_ATTN_TOPK = 3
    DISTANCE_DECAY_COEFF = 0.05
    NUM_FRONT_NODES = 8
    FRONT_THRESHOLD = 0.5
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

# ===================== Sparse Edge Attention =====================
class SparseEdgeAttention(nn.Module):
    """Sparse edge attention module for computing dynamic edge weights."""
    def __init__(self, input_dim: int, num_heads: int = 2, top_k: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.top_k = top_k
        self.head_dim = input_dim // num_heads
        self.q_proj = nn.Linear(input_dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(input_dim, num_heads * self.head_dim)
        self.v_proj = nn.Linear(input_dim, num_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = 1
        num_edges = x.shape[0]
        actual_topk = min(self.top_k, num_edges)
        actual_topk = max(actual_topk, 1)
        q = self.q_proj(x).reshape(batch_size, num_edges, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).reshape(batch_size, num_edges, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).reshape(batch_size, num_edges, self.num_heads, self.head_dim).transpose(1,2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if num_edges <= actual_topk:
            attn_weights = self.softmax(attn_scores)
        else:
            topk_vals, topk_idx = torch.topk(attn_scores, k=actual_topk, dim=-1)
            attn_scores_sparse = torch.full_like(attn_scores, -1e9)
            attn_scores_sparse.scatter_(-1, topk_idx, topk_vals)
            attn_weights = self.softmax(attn_scores_sparse)
        attn_out = torch.matmul(attn_weights, v).transpose(1,2).reshape(batch_size, num_edges, -1)
        edge_weights = self.sigmoid(self.out_proj(attn_out)).squeeze()
        return edge_weights

# ===================== Velocity-Guided Graph Construction =====================
def build_velocity_guided_graph(
    probe_coords: np.ndarray,
    ux_data: np.ndarray,
    uy_data: np.ndarray,
    uz_data: np.ndarray,
    time_step_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Build a velocity‑guided directed graph.

    Args:
        probe_coords: (N, 3) probe coordinates.
        ux_data, uy_data, uz_data: velocity components at each probe for a specific time step.
        time_step_idx: index of the time step to use.

    Returns:
        edge_index: (2, E) edge indices.
        edge_attr: (E, 3) attributes [distance, cos_theta, source speed].
        speed: (N,) speed magnitude.
        radius: (N,) dynamic search radius.
    """
    print("\n" + "="*20 + " Construct Velocity-Guided Directed Graph " + "="*20)
    N = len(probe_coords)
    Ux = ux_data[time_step_idx]
    Uy = uy_data[time_step_idx]
    Uz = uz_data[time_step_idx]
    U = np.stack([Ux, Uy, Uz], axis=1)
    speed = np.linalg.norm(U, axis=1)
    unit_U = np.divide(U, speed[:, None], out=np.zeros_like(U), where=speed[:, None] != 0)

    radius = np.clip(Config.BASE_RADIUS + Config.RADIUS_SCALE_FACTOR * speed, a_min=5.0, a_max=50.0)
    kdtree = KDTree(probe_coords)

    edge_list = []
    edge_attr_list = []

    for src in range(N):
        if speed[src] < Config.SPEED_THRESHOLD:
            continue
        indices = kdtree.query_ball_point(probe_coords[src], r=radius[src])
        indices = [idx for idx in indices if idx != src]
        if len(indices) == 0:
            continue

        rel_pos = probe_coords[indices] - probe_coords[src]
        rel_dist = np.linalg.norm(rel_pos, axis=1) + 1e-8
        unit_rel = rel_pos / rel_dist[:, None]

        cos_theta = np.sum(unit_U[src] * unit_rel, axis=1)
        valid_mask = cos_theta > 0.5
        filtered_idx = np.array(indices)[valid_mask]
        filtered_dist = rel_dist[valid_mask]
        filtered_cos = cos_theta[valid_mask]

        if len(filtered_idx) == 0:
            continue

        scores = filtered_cos * np.exp(-filtered_dist / radius[src])
        top_k = min(Config.KNN_K, len(scores))
        topk_indices = np.argpartition(scores, -top_k)[-top_k:]

        for idx in topk_indices:
            dst = filtered_idx[idx]
            edge_list.append([src, dst])
            edge_attr_list.append([filtered_dist[idx], filtered_cos[idx], speed[src]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(Config.DEVICE)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).to(Config.DEVICE)
    return edge_index, edge_attr, speed, radius

def visualize_spatial_graph(edge_index: torch.Tensor, probe_coords: np.ndarray, edge_weights: torch.Tensor) -> None:
    """Visualize the dynamic spatial graph; edge colors and widths represent weights."""
    G = nx.DiGraph()
    num_nodes = probe_coords.shape[0]
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.T.cpu().numpy()
    G.add_edges_from(edges)
    
    pos = {i: (probe_coords[i, 0], probe_coords[i, 1]) for i in range(num_nodes)}
    plt.figure(figsize=(14, 10))
    
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color="lightblue", alpha=0.8, edgecolors="black")
    
    weights_np = edge_weights.cpu().numpy()
    weights_norm = (weights_np - weights_np.min()) / (weights_np.max() - weights_np.min() + 1e-6)
    edge_colors = plt.cm.Reds(weights_norm)
    edge_widths = 2 + 3 * weights_norm
    
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7,
        arrowstyle='->', arrowsize=15, connectionstyle="arc3,rad=0.05"
    )
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=weights_np.min(), vmax=weights_np.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label("Dynamic Edge Weight (Attention + Distance Decay + Flow Direction)", fontsize=12)
    
    labels = {i: str(i+1) for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")
    
    plt.xlabel("X Coordinate (m)", fontsize=14)
    plt.ylabel("Y Coordinate (m)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "spatial_graph_with_weights.png"), bbox_inches="tight")
    plt.close()
    print("✅ Dynamic spatial graph with edge weights saved to spatial_graph_with_weights.png")

# ===================== Metrics Curve Plotting =====================
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

# ===================== Data Loading and Preprocessing =====================
def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[int], int, np.ndarray, np.ndarray, np.ndarray]:
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
    velocity_dfs = {}
    
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
        
        if feat_name in ["k", "p"]:
            time_vals = df[Config.TIME_COL_NAME].values
            data_array = df[probe_cols].values.astype(np.float32)
            df_long = pd.DataFrame({
                "time_step": np.repeat(time_vals, len(probe_cols)),
                "probe_id_str": np.tile(probe_cols, len(time_vals)).astype(str),
                feat_name: data_array.flatten()
            })
        else:
            df_long = df.melt(
                id_vars=[Config.TIME_COL_NAME], var_name="probe_id_str", value_name=feat_name
            )
            df_long.rename(columns={Config.TIME_COL_NAME: "time_step"}, inplace=True)
        
        df_long["probe_id_str"] = df_long["probe_id_str"].astype(str)
        df_long["probe_id"] = df_long["probe_id_str"].str.replace("", "").astype(int)
        df_long = df_long[df_long["probe_id"].isin(unique_probes)][["time_step", "probe_id", feat_name]]
        
        if feat_name in ["Ux", "Uy", "Uz"]:
            velocity_dfs[feat_name] = df_long
        
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
    
    # No standardization here – return raw data and a placeholder scaler (None)
    scaler = None
    
    all_times = sorted(velocity_dfs["Ux"]["time_step"].unique())
    ux_array = np.zeros((len(all_times), num_probes))
    uy_array = np.zeros((len(all_times), num_probes))
    uz_array = np.zeros((len(all_times), num_probes))
    for i, t in enumerate(all_times):
        t_ux = velocity_dfs["Ux"][velocity_dfs["Ux"]["time_step"] == t].set_index("probe_id")["Ux"]
        t_uy = velocity_dfs["Uy"][velocity_dfs["Uy"]["time_step"] == t].set_index("probe_id")["Uy"]
        t_uz = velocity_dfs["Uz"][velocity_dfs["Uz"]["time_step"] == t].set_index("probe_id")["Uz"]
        for j, pid in enumerate(unique_probes):
            ux_array[i, j] = t_ux.get(pid, 0.0)
            uy_array[i, j] = t_uy.get(pid, 0.0)
            uz_array[i, j] = t_uz.get(pid, 0.0)
    
    print(f"Data loading completed: Feature tensor shape {feat_tensor.shape} (raw, unscaled)")
    return feat_tensor, probe_coords, scaler, unique_probes, num_probes, ux_array, uy_array, uz_array

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

# ===================== DynamicGCN Module =====================
class DynamicGCN(nn.Module):
    """Dynamic Graph Convolutional Network module with two GCN layers and sparse edge attention."""
    def __init__(self, input_dim: int, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.edge_attn = SparseEdgeAttention(
            input_dim=edge_dim + input_dim * 2,
            num_heads=Config.SPARSE_ATTN_HEADS,
            top_k=Config.SPARSE_ATTN_TOPK
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, input_dim = x.shape
        edge_index_batch = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
        edge_attr_batch = edge_attr.unsqueeze(0).repeat(batch_size, 1, 1)
        x_flat = x.reshape(-1, input_dim)
        x_proj = self.input_proj(x_flat)
        
        edge_index_flat = edge_index_batch.reshape(2, -1)
        src_nodes = edge_index_flat[0]
        dst_nodes = edge_index_flat[1]
        edge_attr_flat = edge_attr_batch.reshape(-1, edge_attr.shape[-1])
        edge_input = torch.cat([edge_attr_flat, x_flat[src_nodes], x_flat[dst_nodes]], dim=1)
        
        base_edge_weight = self.edge_attn(edge_input)
        num_edges = edge_attr.shape[0]
        total_flat_edges = num_edges * batch_size
        
        edge_distances = edge_attr_flat[:, 0]
        distance_decay = torch.exp(-Config.DISTANCE_DECAY_COEFF * edge_distances)
        angle_cos = edge_attr_flat[:, 1]
        direction_weight = F.relu(angle_cos)
        
        base_edge_weight = base_edge_weight.squeeze().reshape(total_flat_edges)
        dynamic_weight = base_edge_weight * distance_decay * direction_weight
        dynamic_weight = torch.clamp(dynamic_weight, 0.0, 1.0)
        
        x_gcn1 = self.gcn1(x_flat, edge_index_flat, edge_weight=dynamic_weight)
        x_gcn1 = self.norm1(x_gcn1 + x_proj)
        x_gcn1 = self.dropout(self.relu(x_gcn1))
        
        x_gcn2 = self.gcn2(x_gcn1, edge_index_flat, edge_weight=dynamic_weight)
        x_gcn2 = self.norm2(x_gcn2 + x_gcn1)
        
        return x_gcn2.reshape(batch_size, num_nodes, -1)

# ===================== LSTM Encoder-Decoder + Attention =====================
class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism."""
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.W_encoder = nn.Linear(hidden_dim, attention_dim)
        self.W_decoder = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, input_seq_len, num_nodes, hidden_dim = encoder_outputs.shape
        encoder_proj = self.W_encoder(encoder_outputs)
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)
        attn_scores = self.v(self.tanh(encoder_proj + decoder_proj))
        attn_weights = self.softmax(attn_scores)
        context_vector = torch.sum(encoder_outputs * attn_weights, dim=1)
        return context_vector, attn_weights

class EncoderLSTM(nn.Module):
    """LSTM encoder."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, gcn_seq: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, num_nodes, hidden_dim = gcn_seq.shape
        lstm_input = gcn_seq.permute(0, 2, 1, 3).reshape(-1, seq_len, hidden_dim)
        encoder_outputs, (hidden, cell) = self.lstm(lstm_input)
        encoder_outputs = encoder_outputs.reshape(batch_size, num_nodes, seq_len, hidden_dim).permute(0, 2, 1, 3)
        hidden = hidden.reshape(self.num_layers, batch_size, num_nodes, hidden_dim)
        cell = cell.reshape(self.num_layers, batch_size, num_nodes, hidden_dim)
        return encoder_outputs, (hidden, cell)

class DecoderLSTM(nn.Module):
    """LSTM decoder with attention."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, attention_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attention = BahdanauAttention(hidden_dim, attention_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, target_t: torch.Tensor, encoder_outputs: torch.Tensor,
                hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch_size, num_nodes, input_dim = target_t.shape
        target_t = target_t.reshape(-1, 1, input_dim)
        decoder_hidden = hidden[-1]
        context_vector, attn_weights = self.attention(encoder_outputs, decoder_hidden)
        context_vector = context_vector.reshape(-1, self.hidden_dim)
        lstm_input = torch.cat([target_t, context_vector.unsqueeze(1)], dim=2)
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (
            hidden.reshape(self.num_layers, -1, self.hidden_dim),
            cell.reshape(self.num_layers, -1, self.hidden_dim)
        ))
        hidden = hidden.reshape(self.num_layers, batch_size, num_nodes, self.hidden_dim)
        cell = cell.reshape(self.num_layers, batch_size, num_nodes, self.hidden_dim)
        output = self.fc_out(lstm_out.squeeze(1)).reshape(batch_size, num_nodes, -1)
        return output, (hidden, cell), attn_weights

# ===================== DGCN-Seq2Seq Main Model =====================
class DGCNSeq2Seq(nn.Module):
    """Main model: dynamic GCN encoder + Seq2Seq LSTM + attention decoder."""
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, edge_dim: int,
                 output_dim: int, output_horizon: int, attention_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_horizon = output_horizon
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dynamic_gcn = DynamicGCN(input_dim, hidden_dim, edge_dim)
        self.encoder = EncoderLSTM(hidden_dim, hidden_dim)
        self.decoder = DecoderLSTM(output_dim, hidden_dim, output_dim, attention_dim)
        self.init_decoder_input = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, history_data: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, target_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, input_seq_len, num_nodes, _ = history_data.shape
        gcn_seq = []
        for t in range(input_seq_len):
            x_t = history_data[:, t, :, :]
            gcn_out = self.dynamic_gcn(x_t, edge_index, edge_attr)
            gcn_seq.append(gcn_out)
        gcn_seq = torch.stack(gcn_seq, dim=1)
        
        encoder_outputs, (hidden, cell) = self.encoder(gcn_seq)
        
        decoder_outputs = []
        attn_weights_list = []
        init_input = self.init_decoder_input(gcn_seq[:, -1, :, :])
        current_input = init_input
        
        for t in range(self.output_horizon):
            output, (hidden, cell), attn_weights = self.decoder(
                current_input, encoder_outputs, hidden, cell
            )
            decoder_outputs.append(output)
            attn_weights_list.append(attn_weights)
            if self.training and target_data is not None:
                use_teacher_forcing = torch.rand(1).item() < Config.TEACHER_FORCING_RATIO
                current_input = target_data[:, t, :, :] if use_teacher_forcing else output
            else:
                current_input = output
        
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return decoder_outputs

# ===================== Flood Front Visualization =====================
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
        else:
            velocity_dir_x[t] = 0.0
            velocity_dir_y[t] = 0.0
    
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
    ax1.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax1.set_ylabel('Y Coordinate (m)', fontweight='bold')
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
    ax2.set_xlabel('Time (min)', fontweight='bold')
    ax2.set_ylabel('Instant Velocity (m/min)', fontweight='bold')
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
        ax3.set_xlabel('Velocity (m/min)', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "front_velocity_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    print("✅ The pictures related to flood front and velocity analysis have been saved.")

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
    print("✅ The true vs predicted velocity comparison plot has been saved.")

def plot_true_pred_front_trajectory(true_front: np.ndarray, pred_front: np.ndarray,
                                    probe_coords: np.ndarray) -> None:
    """Create a side‑by‑side comparison of true and predicted front trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.scatter(probe_coords[:, 0], probe_coords[:, 1], color='gray', alpha=0.5, label='Probe Locations')
    ax1.plot(pred_front[:, 0], pred_front[:, 1], color='#A23B72', linewidth=3, marker='s', label='Pred Front')
    for i, (x, y) in enumerate(pred_front[:, :2]):
        ax1.annotate(f'{i*5}min', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')
    ax1.set_title('Pred Flood Front Trajectory (Top View)', fontweight='bold')
    ax1.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax1.set_ylabel('Y Coordinate (m)', fontweight='bold')
    ax1.legend()
    ax1.axis('equal')
    
    ax2.scatter(probe_coords[:, 0], probe_coords[:, 1], color='gray', alpha=0.5, label='Probe Locations')
    ax2.plot(true_front[:, 0], true_front[:, 1], color='#2E86AB', linewidth=3, marker='o', label='True Front')
    for i, (x, y) in enumerate(true_front[:, :2]):
        ax2.annotate(f'{i*5}min', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')
    ax2.set_title('True Flood Front Trajectory (Top View)', fontweight='bold')
    ax2.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax2.set_ylabel('Y Coordinate (m)', fontweight='bold')
    ax2.legend()
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "true_pred_front_trajectory_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Flood front trajectory comparison plot has been saved.")

def plot_pred_flood_front(front_coords: np.ndarray, probe_coords: np.ndarray) -> None:
    """Plot the predicted flood front trajectory alone."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(probe_coords[:, 0], probe_coords[:, 1], color='gray', alpha=0.5, label='Probe Locations')
    ax.plot(front_coords[:, 0], front_coords[:, 1], color='#6A994E', 
            linewidth=3, marker='o', label='Predicted Flood Front Trajectory')
    for i, (x, y) in enumerate(front_coords[:, :2]):
        ax.annotate(f'{i*5}min', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')
    ax.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontweight='bold')
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "pred_flood_front_trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Predicted flood front trajectory plot has been saved.")

# ===================== Training & Validation Visualization =====================
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

# ===================== Training Main Function =====================
def train_and_predict() -> None:
    """Main training pipeline."""
    set_random_seed(42)

    feat_tensor, probe_coords, _, unique_probes, num_probes, ux_array, uy_array, uz_array = load_and_preprocess_data()
    
    edge_index, edge_attr, speed, radius = build_velocity_guided_graph(probe_coords, ux_array, uy_array, uz_array)
    
    train_X, train_Y, val_X, val_Y, test_X, test_Y = create_sliding_window_samples(feat_tensor)
    
    train_flat = train_X.reshape(-1, Config.NODE_FEAT_DIM)
    scaler = StandardScaler()
    scaler.fit(train_flat)
    
    def standardize(data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        orig_shape = data.shape
        flat = data.reshape(-1, Config.NODE_FEAT_DIM)
        scaled = scaler.transform(flat)
        return scaled.reshape(orig_shape)
    
    train_X = standardize(train_X, scaler)
    train_Y = standardize(train_Y, scaler)
    val_X   = standardize(val_X, scaler)
    val_Y   = standardize(val_Y, scaler)
    test_X  = standardize(test_X, scaler)
    test_Y  = standardize(test_Y, scaler)
    
    train_dataset = FloodDataset(train_X, train_Y)
    val_dataset = FloodDataset(val_X, val_Y)
    test_dataset = FloodDataset(test_X, test_Y)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = DGCNSeq2Seq(
        num_nodes=num_probes,
        input_dim=Config.NODE_FEAT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        edge_dim=Config.EDGE_ATTR_DIM,
        output_dim=Config.NODE_FEAT_DIM,
        output_horizon=Config.OUTPUT_WINDOW,
        attention_dim=Config.ATTENTION_DIM
    ).to(Config.DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics_hist = []
    val_metrics_hist = []
    
    print("\n" + "="*20 + " Start Training DGCN-LSTM " + "="*20)
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X, edge_index, edge_attr, batch_Y)
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
                preds = model(batch_X, edge_index, edge_attr)
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'num_nodes': num_probes,
                    'input_dim': Config.NODE_FEAT_DIM,
                    'hidden_dim': Config.HIDDEN_DIM,
                    'edge_dim': Config.EDGE_ATTR_DIM,
                    'output_dim': Config.NODE_FEAT_DIM,
                    'output_horizon': Config.OUTPUT_WINDOW,
                    'attention_dim': Config.ATTENTION_DIM
                },
                'best_val_loss': best_val_loss
            }, os.path.join(RESULT_DIR, "best_dgcn_seq2seq_fixed.pth"))
            print(f"✅ Best model saved (Val Loss: {best_val_loss:.6f})")
    
    plot_training_loss(train_losses, val_losses)
    plot_train_val_metrics(train_metrics_hist, val_metrics_hist)
    
    print("\n" + "="*20 + " Test Phase (Best Model) " + "="*20)
    checkpoint = torch.load(os.path.join(RESULT_DIR, "best_dgcn_seq2seq_fixed.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            preds = model(batch_X, edge_index, edge_attr)
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
    
    pred_front, pred_vel, velocity_metrics = extract_flood_front(test_preds, scaler, probe_coords)
    true_front, true_vel = extract_true_flood_front(test_targets, scaler, probe_coords)

    plot_true_pred_front_trajectory(true_front, pred_front, probe_coords)
    plot_pred_flood_front(pred_front, probe_coords)
    plot_velocity_true_vs_pred(true_vel, pred_vel)
    plot_flood_front_trajectory_with_velocity(pred_front, pred_vel, probe_coords)

    print("✅ Flood front and velocity analysis completed. Results saved to: ", RESULT_DIR)
    
    print("\n" + "="*20 + " Visualize Dynamic Spatial Graph with Weights " + "="*20)
    dgcn_module = model.dynamic_gcn
    dgcn_module.eval()
    dummy_input = torch.tensor(test_X[0:1, 0, :, :], dtype=torch.float32).to(Config.DEVICE)
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    with torch.no_grad():
        src_feat = dummy_input[0, src_nodes]
        dst_feat = dummy_input[0, dst_nodes]
        edge_input = torch.cat([edge_attr, src_feat, dst_feat], dim=1)
        base_weight = dgcn_module.edge_attn(edge_input)
        distance_decay = torch.exp(-Config.DISTANCE_DECAY_COEFF * edge_attr[:, 0])
        dynamic_edge_weights = base_weight * distance_decay * F.relu(edge_attr[:, 1])
        dynamic_edge_weights = torch.clamp(dynamic_edge_weights, 0.0, 1.0)
    visualize_spatial_graph(edge_index, probe_coords, dynamic_edge_weights)
    
    print("\n✅ All tasks completed! Results saved to: ", RESULT_DIR)

# ===================== Entry Point =====================
if __name__ == "__main__":
    try:
        train_and_predict()
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()