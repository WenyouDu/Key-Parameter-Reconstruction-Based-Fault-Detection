import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import FaultDiagnosisSystem, models_config, DATA_CSV_DIR

"""
This file is used to generate scatter plot boundaries and trajectory separation plots.
The threshold lines drawn in the scatter plots represent the drift thresholds of CUSUM, 
i.e., the instantaneous threshold (the minimum deviation required to start accumulation).
The final alarm threshold is not considered here.
"""

plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= Configuration Area =================
NORMAL_FILE = "N_01.csv"
FAULT_FILE = "F_06_A.csv"
GAMMA = 0.986
DRIFT_K_OPT = 0.3
# K value setting for classic CUSUM
DRIFT_K_CLASSIC_SIGMA = 0.4
# ===========================================
# Define uniform plotting range (Zoom Range)
ZOOM_X_LIM = (0, 4.0)  # X-axis: 0-4 (Static Amplitude)
ZOOM_Y_LIM = (0, 4.0)  # Y-axis: 0-4 (Jitter Amplitude)
# ===========================================


class ClassicCUSUM:
    def __init__(self, k_sigma=0.4):
        self.k_sigma = k_sigma
        self.train_std = None
        self.drift = None

    def fit(self, X_train):
        # 1. Calculate normalization parameters
        self.train_std = np.std(X_train, axis=0) + 1e-9

        # 2. Compute features (Absolute value)
        X_norm = X_train / self.train_std
        features = np.abs(X_norm)

        # 3. Calculate global threshold (based on Flattened data)
        feats_flat = features.flatten()
        self.drift = np.mean(feats_flat) + self.k_sigma * np.std(feats_flat)

        print(f"[Classic] Fitted. Threshold (L1) = {self.drift:.4f}")
        return self

    def predict_trajectory(self, X):
        X_norm = X / self.train_std
        features = np.abs(X_norm)

        n = len(features)
        s = np.zeros_like(features)

        # Accumulate independently per channel
        for t in range(1, n):
            val = s[t - 1] + features[t] - self.drift
            s[t] = np.maximum(0, val)

        # Return the maximum value trajectory
        return np.max(s, axis=1)


class OptimizedCUSUM:
    def __init__(self, k_sigma=0.3, gamma=70.0):
        self.k_sigma = k_sigma
        self.gamma = gamma
        self.train_std = None
        self.drift = None

    def fit(self, X_train):
        self.train_std = np.std(X_train, axis=0) + 1e-9

        # Compute features
        X_norm = X_train / self.train_std
        static_e = np.square(X_norm)
        diff_X = np.diff(X_norm, axis=0, prepend=X_norm[0:1, :])
        dynamic_e = np.square(diff_X)
        # features = static_e + self.gamma * dynamic_e
        features = (1 - self.gamma) * static_e + self.gamma * dynamic_e  # Control gamma between 0 and 1

        # Calculate global threshold
        feats_flat = features.flatten()
        self.drift = np.mean(feats_flat) + self.k_sigma * np.std(feats_flat)

        print(f"[Optimized] Fitted. Threshold (Energy) = {self.drift:.4f}")
        return self

    def predict_trajectory(self, X):
        X_norm = X / self.train_std
        static_e = np.square(X_norm)
        diff_X = np.diff(X_norm, axis=0, prepend=X_norm[0:1, :])
        dynamic_e = np.square(diff_X)
        # features = static_e + self.gamma * dynamic_e
        features = (1 - self.gamma) * static_e + self.gamma * dynamic_e  # Control gamma between 0 and 1

        n = len(features)
        s = np.zeros_like(features)

        for t in range(1, n):
            val = s[t - 1] + features[t] - self.drift
            s[t] = np.maximum(0, val)

        return np.max(s, axis=1)


# === Helper Functions ===
def get_residual_features(diagnosis_system, df):
    batch_results = diagnosis_system.get_residuals_batch(df)
    sorted_keys = sorted(batch_results.keys())
    feature_list = []
    for key in sorted_keys:
        res = batch_results[key]['residuals']
        if res.ndim == 1: res = res.reshape(-1, 1)
        feature_list.append(res)
    return np.hstack(feature_list) if feature_list else None


# === Main Program ===
def main():
    print(">>> [Init] Initializing...")
    diagnosis_system = FaultDiagnosisSystem(models_config)

    # 1. Prepare Data
    print(">>> [Data] Reading data...")
    train_files = ["N_01.csv", "N_02.csv", "N_03.csv"]
    X_train_list = []
    for f in train_files:
        try:
            df = pd.read_csv(os.path.join(DATA_CSV_DIR, f)).iloc[501:]
            xr = get_residual_features(diagnosis_system, df)
            if xr is not None: X_train_list.append(xr)
        except:
            pass
    X_train = np.vstack(X_train_list)

    df_n = pd.read_csv(os.path.join(DATA_CSV_DIR, NORMAL_FILE)).iloc[501:]
    xr_n = get_residual_features(diagnosis_system, df_n)
    df_f = pd.read_csv(os.path.join(DATA_CSV_DIR, FAULT_FILE)).iloc[501:]
    xr_f = get_residual_features(diagnosis_system, df_f)
    L = min(len(xr_n), len(xr_f))
    xr_n_plot, xr_f_plot = xr_n[:L], xr_f[:L]

    # 2. Train Models
    classic_model = ClassicCUSUM(k_sigma=DRIFT_K_CLASSIC_SIGMA)
    classic_model.fit(X_train)

    opt_model = OptimizedCUSUM(k_sigma=DRIFT_K_OPT, gamma=GAMMA)
    opt_model.fit(X_train)

    # Obtain global parameters for plotting
    train_std = classic_model.train_std  # Standard deviation is the same for both
    classic_drift_norm = classic_model.drift
    opt_drift = opt_model.drift

    # ================== Figure 1: Scatter Plot Comparison ==================
    print(">>> [Plot] Generating scatter plot...")

    def get_scatter_data(X):
        X_n = X / train_std
        s = np.abs(X_n).flatten()
        diff_X = np.diff(X_n, axis=0, prepend=X_n[0:1, :])
        d = np.abs(diff_X).flatten()
        return s, d

    s_n, d_n = get_scatter_data(xr_n_plot)
    s_f, d_f = get_scatter_data(xr_f_plot)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Common plotting settings ---
    for ax in [ax1, ax2]:
        scatter_f = ax.scatter(s_f, d_f, c='tab:red', alpha=0.6, s=20, label='Fault Points')
        scatter_n = ax.scatter(s_n, d_n, c='tab:blue', alpha=0.6, s=20, label='Normal Points')
        ax.set_xlim(ZOOM_X_LIM)
        ax.set_ylim(ZOOM_Y_LIM)
        ax.set_xlabel(r'Static Deviation ($|e_t|/\sigma$)', fontsize=24)
        ax.set_ylabel(r'Dynamic Differential ($|\Delta e_t|/\sigma$)', fontsize=24)
        ax.grid(True, alpha=0.3)

    # ================= Left Plot: Classic CUSUM =================
    ax1.set_title(f'Classic CUSUM View ($k={DRIFT_K_CLASSIC_SIGMA}$)', fontsize=28)

    # 1. Draw boundary line (Vertical)
    ax1.axvline(x=classic_drift_norm, color='k', linestyle='--', linewidth=2.5)

    # 2. Fill regions
    ax1.fill_betweenx(ZOOM_Y_LIM, 0, classic_drift_norm, color='green', alpha=0.1)
    ax1.fill_betweenx(ZOOM_Y_LIM, classic_drift_norm, ZOOM_X_LIM[1], color='red', alpha=0.1)

    # ================= Right Plot: Optimized CUSUM =================
    ax2.set_title(rf'Optimized CUSUM ($k={DRIFT_K_OPT}$,$\gamma={GAMMA}$)', fontsize=28)

    # 1. Define full X range
    x_all = np.linspace(0, ZOOM_X_LIM[1], 500)

    # 2. Calculate boundary Y values (Energy ellipse: x^2 + gamma*y^2 = drift)
    term_x = (1 - GAMMA) * (x_all ** 2)
    safe_inside = (opt_drift - term_x) / GAMMA
    y_boundary = np.sqrt(np.maximum(0, safe_inside))

    # 3. Fill regions
    ax2.fill_between(x_all, 0, y_boundary, color='green', alpha=0.1, linewidth=0)
    ax2.fill_between(x_all, y_boundary, ZOOM_Y_LIM[1], color='red', alpha=0.1, linewidth=0)

    # 4. Draw black dashed boundary
    mask_curve = y_boundary > 0
    ax2.plot(x_all[mask_curve], y_boundary[mask_curve], color='black', linestyle='--', linewidth=3.5)

    # Legend
    green_patch = mpatches.Patch(color='green', alpha=0.1, label='Safe Zone')
    red_patch = mpatches.Patch(color='red', alpha=0.1, label='Detection Zone')
    boundary_line = Line2D([0], [0], color='black', linewidth=2.5, linestyle='--', label='Decision Boundary')

    for ax in [ax1, ax2]:
        display_handles = [scatter_n, scatter_f, boundary_line, green_patch, red_patch]
        ax.legend(handles=display_handles, loc='upper right', fontsize=18, framealpha=0.9)

    plt.tight_layout()
    save_path1 = 'results/Visual_Analysis/Scatter_Zoomed.png'
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    fig1.savefig(save_path1, dpi=300)

    # ================== Figure 2: Trajectory Comparison (Log Scale) ==================
    print(">>> [Plot] Generating trajectory plot...")

    # Compute trajectories using class methods
    classic_traj_n = classic_model.predict_trajectory(xr_n_plot)
    classic_traj_f = classic_model.predict_trajectory(xr_f_plot)
    opt_traj_n = opt_model.predict_trajectory(xr_n_plot)
    opt_traj_f = opt_model.predict_trajectory(xr_f_plot)

    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # --- 1. Classic Trajectory ---
    ax3.plot(classic_traj_n, label='Normal', color='tab:blue', alpha=0.8, linewidth=1)
    ax3.plot(classic_traj_f, label='Fault', color='tab:red', alpha=0.8, linewidth=1.5)

    # Set Y-axis to Symlog (Symmetric Logarithmic) scale
    ax3.set_yscale('symlog', linthresh=1.0)

    ax3.set_title(f'Trajectory A: Classic CUSUM ($k={DRIFT_K_CLASSIC_SIGMA}$)', fontsize=27)
    ax3.set_ylabel('Score ($S_t$) [Log Scale]', fontsize=24)
    ax3.legend(loc='upper left', fontsize=14)
    # Enable major and minor grid lines for better readability on log scale
    ax3.grid(True, which='both', linestyle=':', alpha=0.6)

    # --- 2. Optimized Trajectory ---
    ax4.plot(opt_traj_n, label='Normal', color='tab:blue', alpha=0.8, linewidth=1)
    ax4.plot(opt_traj_f, label='Fault', color='tab:red', alpha=0.8, linewidth=1.5)

    # Set Y-axis to Symlog scale
    ax4.set_yscale('symlog', linthresh=1.0)

    ax4.set_title(f'Trajectory B: Optimized CUSUM ($k={DRIFT_K_OPT}$,$\gamma={GAMMA}$)', fontsize=24)
    ax4.set_xlabel('Time Step', fontsize=24)
    ax4.set_ylabel('Score ($S_t$) [Log Scale]', fontsize=24)
    ax4.legend(loc='upper left', fontsize=18)
    ax4.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path2 = 'results/Visual_Analysis/Trajectory_Log.png'
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)
    fig2.savefig(save_path2, dpi=300)

    print("\n>>> All tasks completed!")
    print(f"    1. {save_path1}")
    print(f"    2. {save_path2}")


if __name__ == "__main__":
    main()

