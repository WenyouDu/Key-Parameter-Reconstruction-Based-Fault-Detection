import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
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
Perform fault diagnosis using classic methods based on Residuals + Raw Input columns
"""

# Define raw data input columns
CONTEXT_COLS = [1, 10, 11]

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_hybrid_features(diagnosis_system, df):
    """
    Core function: Obtain [Raw Context Features + LSTM Residuals]
    Returns:
        X_combined: Concatenated feature matrix
    """
    # 1. Obtain Residual features
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

    # 3. Feature concatenation (Horizontal Stack)
    # Final Features = [Context, Residuals]
    X_combined = np.hstack([X_ctx, X_res])

    return X_combined


def main():
    # 1. Initialization
    print(">>> [Init] Initializing LSTM Diagnosis System...")
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

            # Get hybrid features
            X_feat = get_hybrid_features(diagnosis_system, df)
            if X_feat is not None:
                X_train_list.append(X_feat)
        except Exception as e:
            print(f"Training file {f} read failed: {e}")

    if not X_train_list:
        print("Training data not obtained, exiting.")
        return

    # Concatenate and standardize
    X_train_raw = np.vstack(X_train_list)
    # fit_transform calculates mean and variance, and transforms the training set
    X_train_scaled = scaler.fit_transform(X_train_raw)

    print(f"Training set feature matrix shape: {X_train_scaled.shape}")
    print(f"  - Context Dim: {len(CONTEXT_COLS)}")
    print(f"  - Residual Dim: {X_train_scaled.shape[1] - len(CONTEXT_COLS)}")

    # 4. Train models
    contamination = 0.001

    classifiers = {
        'Hybrid-KNN': KNN(n_neighbors=5, method='largest', contamination=contamination),
        'Hybrid-LOF': LOF(n_neighbors=20, novelty=True, contamination=contamination),
        'Hybrid-GMM': GMM(n_components=3, covariance_type='full', contamination=contamination),
        'Hybrid-IForest': IForest(n_estimators=100, random_state=42, contamination=contamination),
        'Hybrid-OCSVM': OneClassSVM(kernel='rbf', gamma='scale', nu=0.001),  # Sklearn version
        'Hybrid-PCA': PCA(n_components=None, contamination=contamination),
        'Hybrid-COPOD': COPOD(contamination=contamination),
        'Hybrid-HBOS': HBOS(contamination=contamination),
        'Hybrid-ECOD': ECOD(contamination=contamination),
        'Hybrid-MCD': MCD(contamination=contamination, support_fraction=0.9),
        'Hybrid-ABOD': ABOD(method='fast', n_neighbors=10, contamination=contamination)

    }

    print(f"\n>>> [Phase 2] Training {len(classifiers)} hybrid diagnosis models...")
    for name, clf in classifiers.items():
        print(f"Training {name} ...")
        clf.fit(X_train_scaled)  # Use standardized data

    # 5. Test Evaluation
    print(f"\n>>> [Phase 3] Evaluating on test set...")
    y_true_all = []
    y_scores_all = {name: [] for name in classifiers.keys()}

    for f in test_files:
        path = os.path.join(DATA_CSV_DIR, f)
        label = 0 if f.upper().startswith('N') else 1

        try:
            df = pd.read_csv(path).iloc[501:]
            if df.empty: continue

            # 1. Extract hybrid features
            X_test_feat = get_hybrid_features(diagnosis_system, df)
            if X_test_feat is None: continue

            # 2. Use the trained scaler for standardization
            X_test_scaled = scaler.transform(X_test_feat)

            # Record labels
            y_true_all.append(np.full(len(X_test_scaled), label))

            # 3. Predict
            for name, clf in classifiers.items():
                if 'OCSVM' in name:
                    scores = -clf.decision_function(X_test_scaled)
                else:
                    scores = clf.decision_function(X_test_scaled)
                y_scores_all[name].append(scores)

        except Exception as e:
            print(f"Test file {f} processing failed: {e}")

    # 6. Plotting and Statistics
    if not y_true_all:
        print("No valid test results found.")
        return

    y_true = np.concatenate(y_true_all)
    results = []

    plt.figure(figsize=(14, 12))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(classifiers))]

    for i, name in enumerate(classifiers.keys()):
        if not y_scores_all[name]: continue

        y_score = np.concatenate(y_scores_all[name])
        if np.isnan(y_score).any():
            y_score = np.nan_to_num(y_score)

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        results.append((name, auc, ap))

        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f}, AP={ap:.3f})', color=colors[i])

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Context-Aware Hybrid Diagnosis ROC (Inputs + Residuals)', fontsize=16)
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)

    save_path = 'results/diagnostics_plots/Hybrid_Residual_Raw_Classic_ROC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    results.sort(key=lambda x: x[2], reverse=True)
    print("\n" + "=" * 70)
    print(f"{'Rank':<4} | {'Method':<25} | {'AP (Precision)':<10} | {'AUC (ROC)':<10}")
    print("-" * 75)
    for i, (name, auc, ap) in enumerate(results):
        print(f"{i + 1:<4} | {name:<25} | {ap:.4f}     | {auc:.4f}")
    print("=" * 70)
    print(f"\n[Done] Results saved to: {save_path}")


if __name__ == "__main__":
    main()
