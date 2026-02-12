import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import FaultDiagnosisSystem, models_config, DATA_CSV_DIR

"""
Perform fault diagnosis on residuals using the optimized CUSUM method.
"""

# Set plot style
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AblationCUSUM:
    """
    CUSUM Class
    Allows testing the independent effects of normalization, squaring, and jitter terms
    by parameter toggles.
    Basic logic: S[t] = max(0, S[t-1] + Feature[t] - Drift)
    """
    def __init__(self, drift_k=1.0,
                 use_norm=False,
                 use_square=False,
                 use_jitter=False,
                 jitter_gamma=1.0,
                 name="CUSUM"):
        """
        :param drift_k: Drift coefficient k
        :param use_norm: Whether to normalize residuals by Sigma (to address scale differences between models)
        :param use_square: Whether to use squared features (x^2 vs |x|)
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
        """Training: Calculate statistics and Drift"""
        # 1. Calculate normalization parameter (Sigma)
        if self.use_norm:
            self.train_std = np.std(X, axis=0) + 1e-9
        else:
            self.train_std = np.ones(X.shape[1])

        # 2. Calculate feature stream
        features = self._compute_features(X)

        # 3. Analyze feature distribution to determine Drift
        feat_mean = np.mean(features, axis=0)
        feat_std = np.std(features, axis=0)

        self.drift = feat_mean + self.k_factor * feat_std

        print(f"[{self.name}] Fitted. Drift Avg: {np.mean(self.drift):.6f}")
        return self

    def _compute_features(self, X):
        """Compute features based on toggles"""
        # Step 1: Normalization
        X_processed = X / self.train_std

        # Step 2: Base term
        if self.use_square:
            base_feat = np.square(X_processed)  # x^2
        else:
            base_feat = np.abs(X_processed)  # |x|

        # Step 3: Jitter term
        if self.use_jitter:
            # Calculate first-order difference
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
        """CUSUM accumulation"""
        n_samples, n_features = X.shape
        s = np.zeros((n_samples, n_features))

        # Calculate feature stream
        features = self._compute_features(X)
        drift = self.drift

        for t in range(1, n_samples):
            # CUSUM core formula
            val = s[t - 1] + features[t] - drift
            s[t] = np.maximum(0, val)

        return np.max(s, axis=1)


def get_residual_features(diagnosis_system, df):
    batch_results = diagnosis_system.get_residuals_batch(df)
    sorted_keys = sorted(batch_results.keys())
    feature_list = []
    for key in sorted_keys:
        res = batch_results[key]['residuals']
        if res.ndim == 1:
            res = res.reshape(-1, 1)
        feature_list.append(res)
    return np.hstack(feature_list) if feature_list else None


def main():
    print("Initializing LSTM diagnosis system...")
    diagnosis_system = FaultDiagnosisSystem(models_config)
    if not os.path.exists(DATA_CSV_DIR): return

    train_files = ["N_01.CSV", "N_02.csv", "N_03.csv"]
    all_files = os.listdir(DATA_CSV_DIR)
    test_files = [f for f in all_files if f.endswith('.csv')]
    test_files.sort()

    print(f"\nExtracting training data...")
    X_train_list = []
    for f in train_files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            df = pd.read_csv(path).iloc[501:]
            if not df.empty:
                xr = get_residual_features(diagnosis_system, df)
                if xr is not None: X_train_list.append(xr)
        except:
            pass

    if not X_train_list: return
    X_train = np.vstack(X_train_list)

    # ================== Define experiment groups ==================
    detectors = {
        # Baseline: Original CUSUM (|x|)
        'Baseline (|x|,drift_k=0.5)':
            AblationCUSUM(drift_k=0.5, use_norm=False, use_square=False, use_jitter=False),

        # Individual test: Normalization (to address model scale differences)
        '+Normalization (|x/s|,drift_k=0.5)':
            AblationCUSUM(drift_k=0.5, use_norm=True, use_square=False, use_jitter=False),

        # 3. Individual test: Jitter term (to capture high frequency)
        '+Jitter (|x|+g|dx|, gamma=130.0, drift_k=0.5)':
            AblationCUSUM(drift_k=0.5, use_norm=False, use_square=False, use_jitter=True, jitter_gamma=130.0),

        # Individual test: Squaring (to enhance signal-to-noise ratio)
        '+Squaring (x^2, drift_k=0.5)':
            AblationCUSUM(drift_k=0.5, use_norm=False, use_square=True, use_jitter=False),

        # Combination: Normalization + Squaring (basis for energy detection)
        '+Norm + Square, drift_k=0.5':
            AblationCUSUM(drift_k=0.5, use_norm=True, use_square=True, use_jitter=False),

        # Combination: Normalization + Jitter term (to capture high frequency)
        '+Norm + Jitter, gamma=130.0, drift_k=0.5':
            AblationCUSUM(drift_k=0.5, use_norm=True, use_square=False, use_jitter=True, jitter_gamma=130.0),


        # Combination: Full set (Full Energy)
        '+Norm + Square + Jitter, gamma=0.0, drift_k=0.3':
            AblationCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.0),

        '+Norm + Square + Jitter, gamma=0.5, drift_k=0.3':
            AblationCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=0.5),

        '+Norm + Square + Jitter, gamma=50.0, drift_k=0.3':
            AblationCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=50.0),

        '+Norm + Square + Jitter, gamma=70.0, drift_k=0.3':
            AblationCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=70.0),

        '+Norm + Square + Jitter, gamma=130.0, drift_k=0.3':
            AblationCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=130.0),

        '+Norm + Square + Jitter, gamma=200.0, drift_k=0.3':
            AblationCUSUM(drift_k=0.3, use_norm=True, use_square=True, use_jitter=True, jitter_gamma=200.0),

    }

    print(f"\n>>> [Phase 2] Training models (Ablation Study)...")
    for name, detector in detectors.items():
        detector.fit(X_train)

    # 3. Test
    print(f"\n>>> [Phase 3] Testing and evaluation...")
    y_true_all = []
    y_scores_all = {name: [] for name in detectors.keys()}

    for f in test_files:
        path = os.path.join(DATA_CSV_DIR, f)
        label = 0 if f.upper().startswith('N') else 1
        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue
            xr = get_residual_features(diagnosis_system, df)
            if xr is None: continue

            y_true_all.append(np.full(len(xr), label))
            for name, detector in detectors.items():
                y_scores_all[name].append(detector.decision_function(xr))
        except:
            pass

    # 4. Plotting
    if not y_true_all: return
    y_true = np.concatenate(y_true_all)
    results = []

    plt.figure(figsize=(12, 10))
    # Use jet colormap for clearer contrast
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(detectors))]

    for i, (name, detector) in enumerate(detectors.items()):
        y_score = np.concatenate(y_scores_all[name])
        y_score = np.nan_to_num(y_score)

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        results.append((name, auc, ap))

        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC={auc:.6f})', color=colors[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Ablation Study: Normalization vs Squaring vs Jitter')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_path = 'results/diagnostics_plots/Residual_Optimized_CUSUM_ROC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 100)
    print(f"{'Method (Ablation)':<60} | {'AUC':<15} | {'AP':<15}")
    print("-" * 100)
    for name, auc, ap in results:
        print(f"{name:<60} | {auc:.6f}     | {ap:.6f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
