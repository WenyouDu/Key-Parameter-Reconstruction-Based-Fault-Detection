import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import FaultDiagnosisSystem, models_config, DATA_CSV_DIR

"""
Perform fault diagnosis on the combination of Residuals + Raw Data using the CUSUM method
"""

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Define raw data input columns
CONTEXT_COLS = [1, 10, 11]


class HybridCUSUM:
    """
    CUSUM anomaly detector for hybrid data
    Logic: S_t = max(0, S_{t-1} + |x_t| - drift)
    """
    def __init__(self, drift_k=1.0, name="CUSUM"):
        """
        :param drift_k: Drift coefficient. drift = mean(|x|) + k * std(|x|)
        """
        self.drift_k = drift_k
        self.name = name
        self.drift = None
        self.abs_mean = None

    def fit(self, X):
        """
        Use training data to calibrate the drift parameter (Drift)
        """
        # Calculate absolute value statistics
        abs_X = np.abs(X)
        self.abs_mean = np.mean(abs_X, axis=0)
        abs_std = np.std(abs_X, axis=0)

        # Set the drift threshold
        self.drift = self.abs_mean + self.drift_k * abs_std
        print(f"[{self.name}] Drift parameter fitted.")
        return self

    def decision_function(self, X):
        """
        Calculate CUSUM anomaly scores
        """
        n_samples, n_features = X.shape
        s = np.zeros((n_samples, n_features))

        abs_X = np.abs(X)
        drift = self.drift

        # CUSUM accumulation calculation
        for t in range(1, n_samples):
            val = s[t - 1] + abs_X[t] - drift
            s[t] = np.maximum(0, val)

        # Aggregation: Take the maximum cumulative deviation among all feature dimensions as the system score
        final_scores = np.max(s, axis=1)
        return final_scores


def get_hybrid_features(diagnosis_system, df):
    """
    Core function: Obtain [Raw Context Features + LSTM Residuals]
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
    X_res = np.hstack(res_list)

    # 2. Obtain raw context features
    try:
        X_ctx = df.iloc[:, CONTEXT_COLS].values
    except IndexError:
        print(f"Error: CONTEXT_COLS {CONTEXT_COLS} is out of data column range")
        return None

    # 3. Concatenate
    X_combined = np.hstack([X_ctx, X_res])
    return X_combined


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

    # Initialize scaler
    scaler = StandardScaler()

    # 3. Build training set
    print(f"\n>>> [Phase 1] Extracting training features (Context + Residuals)...")
    X_train_list = []

    for f in train_files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            X_feat = get_hybrid_features(diagnosis_system, df)
            if X_feat is not None:
                X_train_list.append(X_feat)
        except Exception as e:
            print(f"Training file {f} read failed: {e}")

    if not X_train_list: return
    X_train_raw = np.vstack(X_train_list)

    # Standardize training set
    X_train_scaled = scaler.fit_transform(X_train_raw)
    print(f"Training set feature matrix shape: {X_train_scaled.shape}")

    # 4. Define CUSUM model group
    detectors = {
        'CUSUM-Sensitive (k=0.001)': HybridCUSUM(drift_k=0.001, name='Sensitive'),
        'CUSUM-Sensitive (k=0.01)': HybridCUSUM(drift_k=0.01, name='Sensitive'),
        'CUSUM-Sensitive (k=0.1)': HybridCUSUM(drift_k=0.1, name='Sensitive'),
        'CUSUM-Sensitive (k=0.2)': HybridCUSUM(drift_k=0.2, name='Sensitive'),
        'CUSUM-Sensitive (k=0.3)': HybridCUSUM(drift_k=0.3, name='Sensitive'),
        'CUSUM-Sensitive (k=0.4)': HybridCUSUM(drift_k=0.4, name='Sensitive'),
        'CUSUM-Sensitive (k=0.5)': HybridCUSUM(drift_k=0.5, name='Sensitive'),
        'CUSUM-Standard (k=1.0)': HybridCUSUM(drift_k=1.0, name='Standard'),
        'CUSUM-Strict (k=3.0)': HybridCUSUM(drift_k=3.0, name='Strict'),
    }

    print(f"\n>>> [Phase 2] Calibrating CUSUM model parameters...")
    for name, detector in detectors.items():
        detector.fit(X_train_scaled)

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

            # 1. Extract hybrid features
            X_test_feat = get_hybrid_features(diagnosis_system, df)
            if X_test_feat is None: continue

            # 2. Standardize (using parameters from the training set)
            X_test_scaled = scaler.transform(X_test_feat)

            y_true_all.append(np.full(len(X_test_scaled), label))

            # 3. CUSUM prediction
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
    plt.title('Hybrid Context + CUSUM Diagnosis ROC', fontsize=16)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    save_path = 'results/diagnostics_plots/Hybrid_Residual_Raw_CUSUM_ROC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    # Print rankings
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
