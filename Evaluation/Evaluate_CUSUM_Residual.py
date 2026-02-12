import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import FaultDiagnosisSystem, models_config, DATA_CSV_DIR

"""
Perform fault diagnosis on Residuals using the CUSUM method
"""

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ResidualCUSUM:
    """
    CUSUM (Cumulative Sum) anomaly detector for residual data
    Logic: S_t = max(0, S_{t-1} + |x_t| - drift)
    """
    def __init__(self, drift_k=1.0, name="CUSUM"):
        """
        :param drift_k: Drift coefficient. drift = mean(|x|) + k * std(|x|)
                        Higher k means higher noise tolerance (looser threshold), detecting only major faults.
                        Lower k is more sensitive, making it easier to detect micro-drifts, but may be affected by spike noise.
        """
        self.drift_k = drift_k
        self.name = name
        self.drift = None  # Will be calculated in fit
        self.abs_mean = None

    def fit(self, X):
        """
        Use normal training data to calibrate drift parameters (Drift)
        X: (n_samples, n_features) residual matrix
        """
        abs_X = np.abs(X)

        # Calculate mean and standard deviation for each feature dimension
        self.abs_mean = np.mean(abs_X, axis=0)
        abs_std = np.std(abs_X, axis=0)

        self.drift = self.abs_mean + self.drift_k * abs_std

        print(f"[{self.name}] Calibrated Drifts (Avg): {np.mean(self.drift):.6f}")
        return self

    def decision_function(self, X):
        """
        Calculate CUSUM anomaly scores for test data
        """
        n_samples, n_features = X.shape
        s = np.zeros((n_samples, n_features))

        # Get absolute values of residuals
        abs_X = np.abs(X)
        drift = self.drift

        # Iteratively calculate CUSUM
        # S[t] = max(0, S[t-1] + |x[t]| - drift)
        for t in range(1, n_samples):
            # Vectorized calculation for all feature dimensions
            val = s[t - 1] + abs_X[t] - drift
            # Keep only the part greater than 0 (accumulation), otherwise reset to 0
            s[t] = np.maximum(0, val)
        final_scores = np.max(s, axis=1)

        return final_scores


def get_residual_features(diagnosis_system, df):
    """
    Get residual features
    """
    batch_results = diagnosis_system.get_residuals_batch(df)
    sorted_model_keys = sorted(batch_results.keys())

    feature_list = []
    for key in sorted_model_keys:
        res = batch_results[key]['residuals']
        if res.ndim == 1:
            res = res.reshape(-1, 1)
        feature_list.append(res)

    if not feature_list:
        return None

    X_features = np.hstack(feature_list)
    return X_features


def main():
    # 1. Initialization
    print(">>> [Init] Initializing LSTM diagnosis system...")
    diagnosis_system = FaultDiagnosisSystem(models_config)

    if not os.path.exists(DATA_CSV_DIR):
        print(f"Error: Data directory {DATA_CSV_DIR} does not exist")
        return

    # 2. Prepare files
    train_files = ["N_01.CSV", "N_02.csv", "N_03.csv"]
    all_files = os.listdir(DATA_CSV_DIR)
    test_files = [f for f in all_files if f.endswith('.csv')]
    test_files.sort()

    # 3. Build training set
    print(f"\n>>> [Phase 1] Extracting training set residuals (for calibrating CUSUM parameters)...")
    X_train_list = []

    for f in train_files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            X_res = get_residual_features(diagnosis_system, df)
            if X_res is not None:
                X_train_list.append(X_res)
        except Exception as e:
            print(f"Training file {f} read failed: {e}")

    if not X_train_list: return
    X_train = np.vstack(X_train_list)
    print(f"Training set shape: {X_train.shape}")

    # 4. Define CUSUM detector group
    detectors = {
        'CUSUM-Sensitive (k=0.001)': ResidualCUSUM(drift_k=0.001, name='Sensitive'),
        'CUSUM-Sensitive (k=0.01)': ResidualCUSUM(drift_k=0.01, name='Sensitive'),
        'CUSUM-Sensitive (k=0.1)': ResidualCUSUM(drift_k=0.1, name='Sensitive'),
        'CUSUM-Sensitive (k=0.2)': ResidualCUSUM(drift_k=0.2, name='Sensitive'),
        'CUSUM-Sensitive (k=0.3)': ResidualCUSUM(drift_k=0.3, name='Sensitive'),
        'CUSUM-Sensitive (k=0.4)': ResidualCUSUM(drift_k=0.4, name='Sensitive'),
        'CUSUM-Sensitive (k=0.5)': ResidualCUSUM(drift_k=0.5, name='Sensitive'),
        'CUSUM-Balanced (k=1.0)': ResidualCUSUM(drift_k=1.0, name='Balanced'),
        'CUSUM-Robust (k=3.0)': ResidualCUSUM(drift_k=3.0, name='Robust'),
    }

    print(f"\n>>> [Phase 2] Calibrating CUSUM models...")
    for name, detector in detectors.items():
        detector.fit(X_train)

    # 5. Test Evaluation
    print(f"\n>>> [Phase 3] Evaluating on test set...")
    y_true_all = []
    y_scores_all = {name: [] for name in detectors.keys()}

    for f in test_files:
        path = os.path.join(DATA_CSV_DIR, f)
        label = 0 if f.upper().startswith('N') else 1

        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            # 1. Get residuals
            X_test_res = get_residual_features(diagnosis_system, df)
            if X_test_res is None: continue

            y_true_all.append(np.full(len(X_test_res), label))

            # 2. CUSUM Prediction
            for name, detector in detectors.items():
                scores = detector.decision_function(X_test_res)
                y_scores_all[name].append(scores)

        except Exception as e:
            print(f"Test file {f} processing failed: {e}")

    # 6. Plotting
    if not y_true_all: return

    y_true = np.concatenate(y_true_all)
    results = []

    plt.figure(figsize=(12, 10))
    # Colors
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(detectors))]

    for i, (name, detector) in enumerate(detectors.items()):
        if not y_scores_all[name]: continue

        y_score = np.concatenate(y_scores_all[name])

        # Calculate metrics
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        results.append((name, auc, ap))

        # Plot ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC={auc:.3f}, AP={ap:.3f})', color=colors[i])

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Hybrid Diagnosis ROC: LSTM Residuals + CUSUM', fontsize=16)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    save_path = 'results/diagnostics_plots/Residual_CUSUM_ROC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    # Print results
    results.sort(key=lambda x: x[2], reverse=True)
    print("\n" + "=" * 65)
    print(f"{'Method':<25} | {'AP (Precision)':<10} | {'AUC (ROC)':<10}")
    print("-" * 65)
    for name, auc, ap in results:
        print(f"{name:<25} | {ap:.4f}     | {auc:.4f}")
    print("=" * 65)
    print(f"\n[Done] Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
