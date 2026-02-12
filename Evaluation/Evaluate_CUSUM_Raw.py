import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import DATA_CSV_DIR

"""
Fault diagnosis using CUSUM method based on raw data
"""

# SELECTED_COLS = [3, 5, 6, 8, 9, 10, 11]
SELECTED_COLS = [1, 10, 11]

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RawDataCUSUM:
    """
    CUSUM anomaly detector for raw data
    """
    def __init__(self, drift_k=1.0, name="Raw-CUSUM"):
        self.drift_k = drift_k
        self.name = name
        self.train_mean = None  # Record mean
        self.train_std = None  # Record standard deviation (New)
        self.drift = None

    def fit(self, X):
        """
        Training: Calculate mean and standard deviation, and set the drift threshold
        """
        # 1. Calculate statistics
        self.train_mean = np.mean(X, axis=0)
        self.train_std = np.std(X, axis=0) + 1e-9  # Prevent division by zero

        # 2. Standardization (Z-Score)
        X_norm = (X - self.train_mean) / self.train_std

        # 3. Calculate deviation
        deviation = np.abs(X_norm)

        # 4. Set Drift
        dev_mean = np.mean(deviation, axis=0)
        dev_std = np.std(deviation, axis=0)

        self.drift = dev_mean + self.drift_k * dev_std

        print(f"[{self.name}] Fitted. Mean/Std computed. Drift (Normalized): {np.mean(self.drift):.4f}")
        return self

    def decision_function(self, X):
        """
        Predict: Calculate CUSUM anomaly scores
        """
        n_samples, n_features = X.shape

        # 1. During testing, use training set statistics for standardization
        X_norm = (X - self.train_mean) / self.train_std

        # 2. Calculate deviation
        deviation = np.abs(X_norm)

        # 3. CUSUM accumulation
        s = np.zeros((n_samples, n_features))
        drift = self.drift

        for t in range(1, n_samples):
            # Accumulate
            val = s[t - 1] + deviation[t] - drift
            s[t] = np.maximum(0, val)

        # 4. Take the maximum value
        final_scores = np.max(s, axis=1)

        return final_scores


def get_raw_features(df):
    """
    Get raw data features
    """
    selected_cols = SELECTED_COLS

    # Safety check: prevent column index out of bounds
    if df.shape[1] <= max(selected_cols):
        print(f"Warning: Insufficient columns ({df.shape[1]}), cannot select {selected_cols}, using all columns instead.")
        return df.iloc[:, 1:].values

    return df.iloc[:, selected_cols].values


def main():
    print(">>> [Init] Starting CUSUM diagnosis on Raw Data...")

    if not os.path.exists(DATA_CSV_DIR):
        print(f"Error: Data directory {DATA_CSV_DIR} does not exist")
        return

    # 1. Prepare files
    train_files = ["N_01.csv", "N_02.csv", "N_03.csv"]
    all_files = os.listdir(DATA_CSV_DIR)
    test_files = [f for f in all_files if f.endswith('.csv')]
    test_files.sort()

    # 2. Build training set (using raw data)
    print(f"\n>>> [Phase 1] Reading training data (calculating mean and noise floor)...")
    X_train_list = []

    for f in train_files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            # Skip first 501 points (warmup)
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            # Extract raw features directly
            X_raw = get_raw_features(df)
            X_train_list.append(X_raw)
        except Exception as e:
            print(f"File {f} read failed: {e}")

    if not X_train_list: return
    X_train = np.vstack(X_train_list)
    print(f"Training set shape (Raw): {X_train.shape}")

    # 3. Define CUSUM model group
    detectors = {
        'Raw-CUSUM (k=0.001)': RawDataCUSUM(drift_k=0.001, name='Sensitive'),
        'Raw-CUSUM (k=0.01)': RawDataCUSUM(drift_k=0.01, name='Sensitive'),
        'Raw-CUSUM (k=0.1)': RawDataCUSUM(drift_k=0.1, name='Sensitive'),
        'Raw-CUSUM (k=0.2)': RawDataCUSUM(drift_k=0.2, name='Sensitive'),
        'Raw-CUSUM (k=0.3)': RawDataCUSUM(drift_k=0.3, name='Sensitive'),
        'Raw-CUSUM (k=0.4)': RawDataCUSUM(drift_k=0.4, name='Sensitive'),
        'Raw-CUSUM (k=0.5)': RawDataCUSUM(drift_k=0.5, name='Sensitive'),
        'Raw-CUSUM (k=1.0)': RawDataCUSUM(drift_k=1.0, name='Sensitive'),
        'Raw-CUSUM (k=3.0)': RawDataCUSUM(drift_k=3.0, name='Balanced'),
    }

    print(f"\n>>> [Phase 2] Training Raw-CUSUM models...")
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

            # Get test set raw data
            X_test = get_raw_features(df)

            y_true_all.append(np.full(len(X_test), label))

            for name, detector in detectors.items():
                scores = detector.decision_function(X_test)
                y_scores_all[name].append(scores)

        except Exception as e:
            print(f"Error {f}: {e}")

    # 5. Plotting and statistics
    if not y_true_all: return
    y_true = np.concatenate(y_true_all)

    results = []
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(detectors)))

    for i, (name, detector) in enumerate(detectors.items()):
        if not y_scores_all[name]: continue
        y_score = np.concatenate(y_scores_all[name])

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        results.append((name, auc, ap))

        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=2.5, color=colors[i], label=f'{name} (AP={ap:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Baseline: CUSUM on Raw Data (No Residuals)', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_path = 'results/diagnostics_plots/Raw_CUSUM_ROC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    # Print rankings
    results.sort(key=lambda x: x[2], reverse=True)
    print("\n" + "=" * 70)
    print(f"{'Method (Raw Data)':<25} | {'AP (Precision)':<15} | {'AUC (ROC)':<15}")
    print("-" * 70)
    for name, auc, ap in results:
        print(f"{name:<25} | {ap:.4f}          | {auc:.4f}")
    print("=" * 70)
    print(f"\n[Done] Results saved to: {save_path}")


if __name__ == "__main__":
    main()
