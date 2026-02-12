import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import FaultDiagnosisSystem, models_config, DATA_CSV_DIR

"""
Fault diagnosis using Optimized CUSUM method on Residual + Raw Input columns
"""

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Define raw data input columns
CONTEXT_COLS = [1, 10, 11]


class OptimizedCUSUM:
    """
    Optimized CUSUM Class
    """
    def __init__(self, drift_k=1.0,
                 use_norm=False,
                 use_square=False,
                 use_jitter=False,
                 jitter_gamma=1.0,
                 name="CUSUM"):
        """
        :param drift_k: Drift coefficient k
        :param use_norm: Whether to perform additional Sigma normalization on input
        :param use_square: Whether to use squared features (Energy: x^2 vs Amplitude: |x|)
        :param use_jitter: Whether to add a differential jitter term (gamma * |x_t - x_{t-1}|)
        """
        self.k_factor = drift_k
        self.use_norm = use_norm
        self.use_square = use_square
        self.use_jitter = use_jitter
        self.gamma = jitter_gamma
        self.name = name

        self.train_std = None
        self.drift = None

    def fit(self, X):
        """Training: Compute statistics and Drift"""
        # 1. Compute normalization parameters (Sigma)
        if self.use_norm:
            self.train_std = np.std(X, axis=0) + 1e-9
        else:
            self.train_std = np.ones(X.shape[1])

        # 2. Compute feature stream
        features = self._compute_features(X)

        # 3. Statistics of feature distribution to determine Drift
        # Logic: Drift = Mean(Feature) + k * Std(Feature)
        feat_mean = np.mean(features, axis=0)
        feat_std = np.std(features, axis=0)

        self.drift = feat_mean + self.k_factor * feat_std

        print(f"[{self.name}] Fitted. Drift Avg: {np.mean(self.drift):.6f}")
        return self

    def _compute_features(self, X):
        """Compute features based on switches"""
        # Step 1: Internal normalization
        X_processed = X / self.train_std

        # Step 2: Base term
        if self.use_square:
            base_feat = np.square(X_processed)  # Energy form (x^2)
        else:
            base_feat = np.abs(X_processed)  # Amplitude form (|x|)

        # Step 3: Jitter term
        if self.use_jitter:
            # Compute first-order differential
            diff_X = np.diff(X_processed, axis=0, prepend=X_processed[0:1, :])

            if self.use_square:
                jitter_feat = np.square(diff_X)  # (dx)^2
            else:
                jitter_feat = np.abs(diff_X)  # |dx|

            total_feat = base_feat + self.gamma * jitter_feat
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
            # CUSUM Core Formula: S[t] = max(0, S[t-1] + Feature[t] - Drift)
            val = s[t - 1] + features[t] - drift
            s[t] = np.maximum(0, val)

        # Aggregation: Take the maximum value among all feature dimensions as system score
        return np.max(s, axis=1)


def get_hybrid_features(diagnosis_system, df):
    """
    Core function: Obtain [Raw Features + LSTM Residuals]
    """
    # 1. Obtain residual features
    batch_results = diagnosis_system.get_residuals_batch(df)
    sorted_model_keys = sorted(batch_results.keys())

    res_list = []
    for key in sorted_model_keys:
        res = batch_results[key]['residuals']
        if res.ndim == 1:
            res = res.reshape(-1, 1)
        res_list.append(res)

    if not res_list:
        return None

    X_res = np.hstack(res_list)  # (N, n_residuals)

    # 2. Obtain raw data input columns
    try:
        X_ctx = df.iloc[:, CONTEXT_COLS].values
    except IndexError:
        print(f"Error: CONTEXT_COLS {CONTEXT_COLS} is out of data column range")
        return None

    # 3. Feature concatenation: [Context, Residuals]
    X_combined = np.hstack([X_ctx, X_res])

    return X_combined


def main():
    # 1. Initialization
    print("Initializing LSTM diagnosis system")
    diagnosis_system = FaultDiagnosisSystem(models_config)

    if not os.path.exists(DATA_CSV_DIR):
        print(f"Error: Data directory {DATA_CSV_DIR} does not exist")
        return

    # 2. Prepare files
    train_files = ["N_01.CSV", "N_02.csv", "N_03.csv"]
    all_files = os.listdir(DATA_CSV_DIR)
    test_files = [f for f in all_files if f.endswith('.csv')]
    test_files.sort()

    # Initialize external scaler
    scaler = StandardScaler()

    # 3. Build training set
    print(f"\nExtracting training features (Context + Residuals)")
    X_train_list = []

    for f in train_files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            # Get hybrid features
            X_feat = get_hybrid_features(diagnosis_system, df)
            if X_feat is not None:
                X_train_list.append(X_feat)
        except Exception as e:
            print(f"Training file {f} read failed: {e}")

    if not X_train_list: return

    # Concatenate and standardize
    X_train_raw = np.vstack(X_train_list)
    X_train_scaled = scaler.fit_transform(X_train_raw)

    print(f"Training set feature matrix shape: {X_train_scaled.shape}")
    print(f"  (Contains {len(CONTEXT_COLS)} columns of raw data + {X_train_scaled.shape[1] - len(CONTEXT_COLS)} columns of residuals)")

    # 4. Define Optimized CUSUM model group

    detectors = {
        'Full Optimized (drift_k=0.3, x^2 + Jitter=0.0)':
            OptimizedCUSUM(drift_k=0.3, use_norm=False, use_square=True, use_jitter=True, jitter_gamma=0.),

        'Full Optimized (drift_k=0.3, x^2 + Jitter=0.5)':
            OptimizedCUSUM(drift_k=0.3, use_norm=False, use_square=True, use_jitter=True, jitter_gamma=0.5),

        'Full Optimized (rift_k=0.3, x^2 + Jitter=50.0)':
            OptimizedCUSUM(drift_k=0.3, use_norm=False, use_square=True, use_jitter=True, jitter_gamma=50.0),

        'Full Optimized (rift_k=0.3, x^2 + Jitter=70.0)':
            OptimizedCUSUM(drift_k=0.3, use_norm=False, use_square=True, use_jitter=True, jitter_gamma=70.0),

        'Full Optimized (drift_k=0.3, x^2 + Jitter=130)':
            OptimizedCUSUM(drift_k=0.3, use_norm=False, use_square=True, use_jitter=True, jitter_gamma=130),

        'Full Optimized (drift_k=0.3, x^2 + Jitter=140)':
            OptimizedCUSUM(drift_k=0.3, use_norm=False, use_square=True, use_jitter=True, jitter_gamma=140),

        'Full Optimized (drift_k=0.3, x^2 + Jitter=200)':
            OptimizedCUSUM(drift_k=0.3, use_norm=False, use_square=True, use_jitter=True, jitter_gamma=200),

    }

    print(f"\nTraining Optimized CUSUM models")
    for name, detector in detectors.items():
        detector.fit(X_train_scaled)

    # 5. Test Evaluation
    print(f"\nEvaluating on test set")
    y_true_all = []
    y_scores_all = {name: [] for name in detectors.keys()}

    for f in test_files:
        path = os.path.join(DATA_CSV_DIR, f)
        label = 0 if f.upper().startswith('N') else 1

        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            # 1. Extract hybrid features
            X_test_feat = get_hybrid_features(diagnosis_system, df)
            if X_test_feat is None: continue

            # 2. Standardize (using parameters from the training set)
            X_test_scaled = scaler.transform(X_test_feat)

            y_true_all.append(np.full(len(X_test_scaled), label))

            # 3. Predict
            for name, detector in detectors.items():
                scores = detector.decision_function(X_test_scaled)
                y_scores_all[name].append(scores)

        except Exception as e:
            print(f"Test file {f} processing failed: {e}")

    # 6. Plotting and Statistics
    if not y_true_all: return
    y_true = np.concatenate(y_true_all)
    results = []

    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(detectors))]

    for i, (name, detector) in enumerate(detectors.items()):
        if not y_scores_all[name]: continue

        y_score = np.concatenate(y_scores_all[name])
        y_score = np.nan_to_num(y_score)

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        results.append((name, auc, ap))

        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC={auc:.4f})', color=colors[i])

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Hybrid Context + Optimized CUSUM Diagnosis', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    save_path = 'results/diagnostics_plots/Hybrid_Residual_Raw_Optimized_CUSUM_ROC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    # Print results
    results.sort(key=lambda x: x[2], reverse=True)
    print("\n" + "=" * 75)
    print(f"{'Method':<35} | {'AP (Precision)':<15} | {'AUC (ROC)':<15}")
    print("-" * 75)
    for name, auc, ap in results:
        print(f"{name:<35} | {ap:.6f}          | {auc:.6f}")
    print("=" * 75)
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    main()
