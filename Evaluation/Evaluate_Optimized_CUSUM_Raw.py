import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import DATA_CSV_DIR

"""
Fault diagnosis using Optimized CUSUM method based on Raw Data (gamma set between 0 and 1)
"""

# ================= Configuration Area =================
# SELECTED_COLS = [3, 5, 6, 8, 9, 10, 11]
SELECTED_COLS = [1, 10, 11]

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RawOptimizedCUSUM:
    """
    Optimized CUSUM class for raw data
    """
    def __init__(self, drift_k=1.0,
                 use_norm=False,
                 use_square=False,
                 use_jitter=False,
                 jitter_gamma=1.0,
                 name="CUSUM"):
        """
        :param drift_k: Drift coefficient k
        :param use_norm: Whether to perform Sigma normalization (Z-Score Scaling)
        :param use_square: Whether to use squared features (Energy: x^2) vs Amplitude (|x|)
        :param use_jitter: Whether to add a differential jitter term (High-freq: |dx|)
        :param jitter_gamma: Weight for the jitter term
        """
        self.k_factor = drift_k
        self.use_norm = use_norm
        self.use_square = use_square
        self.use_jitter = use_jitter
        self.gamma = jitter_gamma
        self.name = name

        self.train_mean = None
        self.train_std = None
        self.drift = None

    def fit(self, X):
        """Training: Calculate mean, standard deviation, and compute Drift"""
        # 1. Statistics for center
        self.train_mean = np.mean(X, axis=0)

        # 2. Statistics for standard deviation (used for normalization)
        if self.use_norm:
            self.train_std = np.std(X, axis=0) + 1e-9
        else:
            self.train_std = np.ones(X.shape[1])

        # 3. Compute feature stream
        features = self._compute_features(X)

        # 4. Statistics for feature distribution
        feat_mean = np.mean(features, axis=0)
        feat_std = np.std(features, axis=0)

        # Adaptive threshold formula
        self.drift = feat_mean + self.k_factor * feat_std

        print(f"[{self.name}] Fitted. Drift Avg: {np.mean(self.drift):.6f}")
        return self

    def _compute_features(self, X):
        """Compute optimized features (Core Algorithm)"""
        # Step 1: Centralization + Normalization
        X_centered = X - self.train_mean
        X_processed = X_centered / self.train_std

        # Step 2: Base term
        if self.use_square:
            base_feat = np.square(X_processed)  # Energy form (x^2)
        else:
            base_feat = np.abs(X_processed)  # Amplitude form (|x|)

        # Step 3: Jitter term
        if self.use_jitter:
            # Compute first-order differential (describes high-frequency changes)
            diff_X = np.diff(X_processed, axis=0, prepend=X_processed[0:1, :])

            if self.use_square:
                jitter_feat = np.square(diff_X)  # (dx)^2
            else:
                jitter_feat = np.abs(diff_X)  # |dx|

            # total_feat = base_feat + self.gamma * jitter_feat
            total_feat = (1 - self.gamma) * base_feat + self.gamma * jitter_feat  # Control gamma between 0 and 1
        else:
            total_feat = base_feat

        return total_feat

    def decision_function(self, X):
        """CUSUM Accumulation"""
        n_samples, n_features = X.shape
        s = np.zeros((n_samples, n_features))

        # Compute feature stream
        features = self._compute_features(X)
        drift = self.drift

        for t in range(1, n_samples):
            # CUSUM core recursive formula
            val = s[t - 1] + features[t] - drift
            s[t] = np.maximum(0, val)

        # Aggregation: Take the maximum cumulative value among all dimensions
        return np.max(s, axis=1)


def get_raw_features(df):
    """
    Obtain raw data features (kept unchanged)
    Only select column indices [1, 10, 11] (Cmd_Pos, Current, Velocity)
    """
    selected_cols = SELECTED_COLS
    if df.shape[1] <= max(selected_cols):
        return df.iloc[:, 1:].values
    return df.iloc[:, selected_cols].values


def main():
    print(">>> [Init] Starting Optimized CUSUM diagnosis on Raw Data...")

    if not os.path.exists(DATA_CSV_DIR):
        print(f"Error: Data directory {DATA_CSV_DIR} does not exist")
        return

    # 1. Prepare files
    train_files = ["N_01.CSV", "N_02.csv", "N_03.csv"]
    all_files = os.listdir(DATA_CSV_DIR)
    test_files = [f for f in all_files if f.endswith('.csv')]
    test_files.sort()

    # 2. Build training set
    print(f"\n>>> [Phase 1] Reading training data...")
    X_train_list = []
    for f in train_files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue
            X_raw = get_raw_features(df)
            X_train_list.append(X_raw)
        except Exception as e:
            print(f"File {f} read failed: {e}")

    if not X_train_list: return
    X_train = np.vstack(X_train_list)
    print(f"Training set shape (Raw): {X_train.shape}")

    # 3. Define Optimized CUSUM model group
    # detectors = {
    #     '+Square + Jitter (g=0.0, k=0.3)':
    #         RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.0),
    #
    #     '+Square + Jitter (g=0.5, k=0.3)':
    #         RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.5),
    #
    #     '+Square + Jitter (g=50, k=0.3)':
    #         RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=50.0),
    #
    #     '+Square + Jitter (g=70, k=0.3)':
    #         RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=70.0),
    #
    #     '+Square + Jitter (g=130, k=0.3)':
    #         RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=130.0),
    #
    #     '+Square + Jitter (g=200, k=0.3)':
    #         RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=200.0),
    #
    # }
    detectors = {
        # Baseline: No jitter (gamma=0)
        'Optimized CUSUM (g=0.00)':
            RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.0),

        # Original g=0.5 -> 0.5/1.5 = 0.333
        'Optimized CUSUM (g=0.33)':
            RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.333),

        # Original g=50 -> 50/51 = 0.9804
        'Optimized CUSUM (g=0.980)':
            RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.9804),

        # Original g=70 -> 70/71 = 0.9859
        'Optimized CUSUM (g=0.986)':
            RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.9859),

        # Original g=130 -> 130/131 = 0.9924
        'Optimized CUSUM (g=0.992)':
            RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.9924),

        # Original g=200 -> 200/201 = 0.9950
        'Optimized CUSUM (g=0.995)':
            RawOptimizedCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.9950),
    }

    print(f"\n>>> [Phase 2] Training Optimized Models (Ablation Study)...")
    for name, detector in detectors.items():
        detector.fit(X_train)

    # 4. Test Evaluation
    print(f"\n>>> [Phase 3] Full test evaluation...")
    y_true_all = []
    y_scores_all = {name: [] for name in detectors.keys()}

    for f in test_files:
        path = os.path.join(DATA_CSV_DIR, f)
        label = 0 if f.upper().startswith('N') else 1
        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            X_test = get_raw_features(df)
            y_true_all.append(np.full(len(X_test), label))

            for name, detector in detectors.items():
                scores = detector.decision_function(X_test)
                y_scores_all[name].append(scores)
        except Exception as e:
            pass

    # 5. Plotting and Statistics
    if not y_true_all: return
    y_true = np.concatenate(y_true_all)
    results = []

    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(detectors))]

    for i, (name, detector) in enumerate(detectors.items()):
        if not y_scores_all[name]: continue
        y_score = np.concatenate(y_scores_all[name])
        y_score = np.nan_to_num(y_score)  # Handle possible NaNs

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        results.append((name, auc, ap))

        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC={auc:.4f})', color=colors[i])

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Optimized CUSUM on Raw Data (Ablation Study)', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_path = 'results/diagnostics_plots/Raw_Data_Optimized_CUSUM_ROC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 80)
    print(f"{'Method':<40} | {'AUC':<10} | {'AP':<10}")
    print("-" * 80)
    for name, auc, ap in results:
        print(f"{name:<40} | {auc:.4f}     | {ap:.4f}")
    print("=" * 80)
    print(f"\n[Done] Results saved to: {save_path}")


if __name__ == "__main__":
    main()

