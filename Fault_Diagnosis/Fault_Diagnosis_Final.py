import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import FaultDiagnosisSystem, models_config, DATA_CSV_DIR

"""
This file uses the optimized CUSUM algorithm based on component models to perform fault diagnosis on the full dataset,
and generates summary plots including anomaly score accumulation, injection timing, and decision thresholds.
"""

# ================= Configuration Parameters =================
# Optimized CUSUM Parameters
DRIFT_K = 0.3  # Drift coefficient k
GAMMA = 0.333  # Jitter term weight gamma
SAFETY_FACTOR = 1.0  # Safety factor eta

# Plotting Settings
PLOT_SAVE_DIR = os.path.join('results', 'diagnosis_results_optimized_cusum')
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Global Font Settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


class OptimizedCUSUMDetector:
    """
    Fault detector based on the optimized CUSUM algorithm
    """
    def __init__(self, drift_k=0.5, gamma=1.0, safety_factor=1.2):
        self.k = drift_k
        self.gamma = gamma
        self.eta = safety_factor
        self.sigma = None
        self.drift = None
        self.threshold = None

    def fit(self, normal_residuals_list):
        print(f"[-] Fitting Optimized CUSUM Model (K={self.k}, Gamma={self.gamma}, Eta={self.eta})...")
        # 1. Calculate Sigma
        all_res_concat = np.vstack(normal_residuals_list)
        self.sigma = np.std(all_res_concat, axis=0) + 1e-9

        # 2. Calculate Drift
        all_features = []
        for res in normal_residuals_list:
            feat = self._compute_instant_feature(res)
            all_features.append(feat)
        all_features_concat = np.vstack(all_features)
        feat_mean = np.mean(all_features_concat, axis=0)
        feat_std = np.std(all_features_concat, axis=0)
        self.drift = feat_mean + self.k * feat_std

        # 3. Calculate Threshold
        max_s_values = []
        for res in normal_residuals_list:
            s_seq = self._compute_cusum_statistic(res)
            max_s_values.append(np.max(s_seq))

        global_max_s_normal = np.max(max_s_values)
        self.threshold = self.eta * global_max_s_normal

        print(f" Threshold (J_th): {self.threshold:.4f}")

    def _compute_instant_feature(self, residuals):
        norm_res = residuals / self.sigma
        energy_term = np.square(norm_res)
        diff_res = np.diff(norm_res, axis=0, prepend=norm_res[0:1, :])
        jitter_term = np.square(diff_res)
        return (1 - self.gamma) * energy_term + self.gamma * jitter_term

    def _compute_cusum_statistic(self, residuals):
        n_samples, n_features = residuals.shape
        features = self._compute_instant_feature(residuals)
        s = np.zeros_like(features)
        for t in range(1, n_samples):
            val = s[t - 1] + features[t] - self.drift
            s[t] = np.maximum(0, val)
        return np.max(s, axis=1)

    def diagnose(self, residuals):
        if self.threshold is None: raise ValueError("Model has not been fitted")
        s_score = self._compute_cusum_statistic(residuals)
        is_fault_sequence = s_score > self.threshold
        return s_score, is_fault_sequence


def get_combined_residuals(diagnosis_system, df):
    batch_results = diagnosis_system.get_residuals_batch(df)
    sorted_keys = sorted(batch_results.keys())
    res_list = []
    for key in sorted_keys:
        res = batch_results[key]['residuals']
        if res.ndim == 1: res = res.reshape(-1, 1)
        res_list.append(res)
    if not res_list: return None
    return np.hstack(res_list)


def plot_grid_results(results_list, global_threshold, save_dir):
    """
        Plot 9x4 Grid - Including Fault Injection Timing Markers
    """
    n_samples = len(results_list)

    n_cols = 4
    n_rows = 9
    if n_samples > 36:
        n_rows = math.ceil(n_samples / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), dpi=300, sharex=True, sharey=False)
    axes_flat = axes.flatten()

    # Sampling rate setting
    FS = 100

    for i, ax in enumerate(axes_flat):
        if i < n_samples:
            res_data = results_list[i]
            s_score = res_data['score']
            label_name = os.path.splitext(res_data['filename'])[0]
            time_steps = np.arange(len(s_score))

            # 1. Plotting
            ax.plot(time_steps, s_score, color='#1f77b4', linewidth=1.2)

            # Threshold Line
            ax.axhline(global_threshold, color='red', linestyle='--', linewidth=1.8, alpha=0.9)

            # Fill Red Alarm Area
            ax.fill_between(time_steps, s_score, global_threshold,
                            where=(s_score > global_threshold),
                            color='red', alpha=0.3, interpolate=True)
            # ================= Plot Fault Injection Timing Vertical Line =================
            injection_step = None

            # Marker only for fault files (starting with F)
            if label_name.upper().startswith('F'):
                if '01' in label_name:
                    injection_step = 180 * FS
                elif '13' in label_name:
                    injection_step = 500 * FS
                else:
                    injection_step = 0

            if injection_step is not None:
                # Plot black dash-dot line
                ax.axvline(x=injection_step, color='black', linestyle='-.', linewidth=1.8, alpha=0.8)
            # 2. Set Symlog
            ax.set_yscale('symlog', linthresh=1.0)

            # 3. Set Y-axis Range
            current_max = np.max(s_score)
            upper_lim = max(current_max, global_threshold) * 10.0
            ax.set_ylim(bottom=0.0, top=upper_lim)

            if upper_lim > 1:
                max_exp = int(np.log10(upper_lim))
            else:
                max_exp = 1

            all_exponents = np.arange(0, max_exp + 1)

            if len(all_exponents) > 3:
                target_indices = np.linspace(0, len(all_exponents) - 1, 4, dtype=int)
                target_indices = np.unique(target_indices)
                selected_exponents = all_exponents[target_indices]

                major_ticks = [10 ** int(e) for e in selected_exponents]
            else:
                major_ticks = [10 ** int(e) for e in all_exponents]

            # Apply major ticks
            ax.set_yticks(major_ticks)

            # Minor grid lines
            minor_ticks = [10 ** int(e) for e in all_exponents]
            ax.set_yticks(minor_ticks, minor=True)

            # Left-align Y-axis labels
            ax.tick_params(axis='y', pad=15, labelsize=13)
            plt.setp(ax.get_yticklabels(), ha='left')

            # 4. Label and Decoration
            ax.text(0.95, 0.05, label_name, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', horizontalalignment='right', verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            ax.tick_params(axis='x', labelsize=10)

            if i // n_cols == n_rows - 1:
                ax.set_xlabel("Time step", fontsize=18, fontfamily='serif')

            ax.grid(True, which='major', linestyle='-', alpha=0.4, color='gray')
            ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray')

        else:
            ax.axis('off')

    # Global Y-axis Label
    fig.text(0.01, 0.5, 'Anomaly Score ($S_t$) [Log Scale]', va='center', rotation='vertical', fontsize=22,
             fontfamily='serif')

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.98])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    save_path = os.path.join(save_dir, 'Global_Diagnosis_Grid_NoZero.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Optimized plot saved to: {save_path}")


def main():
    # 1. Initialize Diagnosis System
    print(">>> Initializing LSTM Model System")
    diagnosis_system = FaultDiagnosisSystem(models_config)

    if not os.path.exists(DATA_CSV_DIR):
        print(f"Data directory does not exist: {DATA_CSV_DIR}")
        return

    all_files = os.listdir(DATA_CSV_DIR)
    # Normal files (for training threshold)
    train_files = ["N_01.CSV", "N_02.csv", "N_03.csv"]

    # Test files
    test_files = [f for f in all_files if f.endswith('.csv')]
    test_files.sort()

    # Print file list to confirm order (first 26 should start with F, last 10 should start with N)
    print(f"Detected a total of {len(test_files)} test files.")
    if len(test_files) >= 1: print(f"First file: {test_files[0]}")
    if len(test_files) >= 26: print(f"26th file: {test_files[25]}")

    # --- Phase 1: Fit Threshold ---
    print(f"\nFitting Threshold")
    normal_residuals_list = []
    for f in train_files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue
            res = get_combined_residuals(diagnosis_system, df)
            if res is not None: normal_residuals_list.append(res)
        except Exception:
            pass

    if not normal_residuals_list: return

    detector = OptimizedCUSUMDetector(drift_k=DRIFT_K, gamma=GAMMA, safety_factor=SAFETY_FACTOR)
    detector.fit(normal_residuals_list)

    # --- Phase 2: Batch Diagnosis ---
    print(f"\nBatch Diagnosing All Files (Files: {len(test_files)})")

    all_diagnosis_results = []

    for f in test_files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            res = get_combined_residuals(diagnosis_system, df)
            if res is None: continue

            s_score, is_fault = detector.diagnose(res)

            all_diagnosis_results.append({
                'filename': f,
                'score': s_score
            })

            max_s = np.max(s_score)
            status = "FAULT" if np.any(is_fault) else "NORMAL"
            print(f"  -> {f:<15} | {status} | Max S: {max_s:.2e}")  # Use scientific notation for print

        except Exception as e:
            print(f"Diagnosis error for {f}: {e}")

    # --- Phase 3: Plotting ---
    if all_diagnosis_results:
        plot_grid_results(all_diagnosis_results, detector.threshold, PLOT_SAVE_DIR)


if __name__ == "__main__":
    main()