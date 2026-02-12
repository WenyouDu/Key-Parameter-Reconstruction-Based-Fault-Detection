import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.svm import OneClassSVM
# ================= Import various classic models from PyOD library =================
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.gmm import GMM
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.mcd import MCD
from pyod.models.abod import ABOD
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import FaultDiagnosisSystem, models_config, DATA_CSV_DIR

"""
Perform fault diagnosis on residuals using classic methods
"""

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_residual_features(diagnosis_system, df):
    """
    Core function: Obtain residuals from the LSTM system and concatenate them into feature vectors
    Returns shape: (n_samples, n_features)
    """
    batch_results = diagnosis_system.get_residuals_batch(df)

    # Ensure features are extracted in a fixed order
    sorted_model_keys = sorted(batch_results.keys())

    feature_list = []
    for key in sorted_model_keys:
        # Obtain raw residuals
        res = batch_results[key]['residuals']

        # If it is a 1D array (N,), convert to (N, 1)
        if res.ndim == 1:
            res = res.reshape(-1, 1)

        feature_list.append(res)

    if not feature_list:
        return None

    X_features = np.hstack(feature_list)
    return X_features


def main():
    # 1. Initialize LSTM diagnosis system
    print(">>> [Init] Initializing LSTM diagnosis system...")
    diagnosis_system = FaultDiagnosisSystem(models_config)

    if not os.path.exists(DATA_CSV_DIR):
        print(f"Error: Data directory {DATA_CSV_DIR} does not exist")
        return

    # 2. Prepare file lists
    train_files = ["N_01.CSV", "N_02.csv", "N_03.csv"]

    # Test set: All CSV files in the directory
    all_files = os.listdir(DATA_CSV_DIR)
    test_files = [f for f in all_files if f.endswith('.csv')]
    test_files.sort()

    # 3. Build training set
    print(f"\n>>> [Phase 1] Extracting training set residuals (for training KNN/LOF/ECOD/MCD...)...")
    X_train_list = []

    for f in train_files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            # Read data (Skip the first 501 warmup points)
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            # Extract residual features
            X_res = get_residual_features(diagnosis_system, df)
            if X_res is not None:
                X_train_list.append(X_res)
        except Exception as e:
            print(f"Training file {f} read failed: {e}")

    if not X_train_list:
        print("Training data not obtained, exiting.")
        return

    # Concatenate residuals from all training files
    X_train = np.vstack(X_train_list)
    print(f"Training set feature matrix shape (LSTM Residuals): {X_train.shape}")

    # 4. Define and train PyOD models + Sklearn OCSVM
    # contamination is set very small because we use clean data for training (Normal Only)
    contamination = 0.001

    classifiers = {
        # --- Classic Methods ---
        'Residual-KNN': KNN(n_neighbors=5, method='largest', contamination=contamination),
        'Residual-LOF': LOF(n_neighbors=20, novelty=True, contamination=contamination),
        'Residual-GMM': GMM(n_components=3, covariance_type='full', contamination=contamination),
        'Residual-IForest': IForest(n_estimators=100, random_state=42, contamination=contamination),

        # --- Using Sklearn Native OneClassSVM (faster) ---
        # nu=0.001 corresponds to contamination, indicating a very low proportion of anomalies allowed in the training set
        'Residual-OCSVM': OneClassSVM(kernel='rbf', gamma='scale', nu=0.001),

        'Residual-PCA': PCA(n_components=None, contamination=contamination),
        'Residual-COPOD': COPOD(contamination=contamination),
        'Residual-HBOS': HBOS(contamination=contamination),
        'Residual-ECOD': ECOD(contamination=contamination),
        'Residual-MCD': MCD(contamination=contamination, support_fraction=0.9),
        'Residual-ABOD': ABOD(method='fast', n_neighbors=10, contamination=contamination)
    }

    print(f"\n>>> [Phase 2] Training {len(classifiers)} secondary classifiers...")
    for name, clf in classifiers.items():
        print(f"Training {name} ...")
        clf.fit(X_train)

    # 5. Build test set and predict
    print(f"\n>>> [Phase 3] Evaluating on test set...")
    y_true_all = []
    # Store scores for each model across all test data
    y_scores_all = {name: [] for name in classifiers.keys()}

    for f in test_files:
        path = os.path.join(DATA_CSV_DIR, f)
        # Label: Starting with 'N' is 0 (Normal), otherwise 1 (Anomaly)
        label = 0 if f.upper().startswith('N') else 1

        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            # 1. Extract residuals using LSTM
            X_test_res = get_residual_features(diagnosis_system, df)
            if X_test_res is None: continue

            # Record labels
            y_true_all.append(np.full(len(X_test_res), label))

            # Prevent cases where variance is 0
            if np.any(np.std(X_test_res, axis=0) == 0):
                X_test_res += 1e-9

            # 2. Predict anomaly scores using models
            for name, clf in classifiers.items():
                if name == 'Residual-OCSVM':
                    # Sklearn OCSVM's decision_function returns "distance"
                    # Positive = Normal, Negative = Anomaly.
                    # To get "anomaly score" (higher is more anomalous), take the negative
                    scores = -clf.decision_function(X_test_res)
                else:
                    # PyOD models directly return anomaly scores (higher is more anomalous)
                    scores = clf.decision_function(X_test_res)

                y_scores_all[name].append(scores)

        except Exception as e:
            print(f"Test file {f} processing failed: {e}")

    # 6. Aggregate results and plot
    if not y_true_all:
        print("No valid test results found.")
        return

    y_true = np.concatenate(y_true_all)

    results = []

    plt.figure(figsize=(14, 12))

    # Use colormap to generate enough colors
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(classifiers))]

    # Iterate and plot
    for i, name in enumerate(classifiers.keys()):
        if not y_scores_all[name]:
            continue

        y_score = np.concatenate(y_scores_all[name])

        # Handle potential NaNs
        if np.isnan(y_score).any():
            y_score = np.nan_to_num(y_score)

        # Calculate metrics
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        results.append((name, auc, ap))

        # Plot ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f}, AP={ap:.3f})', color=colors[i])

    # Plot random guess baseline
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Guess')

    # Decorate chart
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Comprehensive Hybrid Diagnosis ROC (10 Methods)', fontsize=16)
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)

    # Save plot
    save_path = 'results/diagnostics_plots/Residual_Classic_ROC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    # 7. Print Rankings
    results.sort(key=lambda x: x[2], reverse=True)  # Sort by AP
    print("\n" + "=" * 70)
    print(f"{'Rank':<4} | {'Method':<20} | {'AP (Precision)':<10} | {'AUC (ROC)':<10}")
    print("-" * 70)
    for i, (name, auc, ap) in enumerate(results):
        print(f"{i + 1:<4} | {name:<20} | {ap:.4f}     | {auc:.4f}")
    print("=" * 70)
    print(f"\n[Done] Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
