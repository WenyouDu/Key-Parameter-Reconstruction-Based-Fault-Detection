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
from Hydraulic_Control_Surface_Fault_Detection.Fault_Diagnosis.Faults_Diagnose import DATA_CSV_DIR

"""
Perform fault diagnosis on raw data using classic methods
"""

# Feature columns for raw data
FEATURE_COLS = [3, 5, 6, 8, 9, 10, 11]
# FEATURE_COLS = [1, 10, 11]

# Plotting settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_raw_data(files, scaler=None, fit_scaler=False):
    """
    Read raw data and perform standardization
    """
    data_list = []
    for f in files:
        path = os.path.join(DATA_CSV_DIR, f)
        try:
            # Read data (Skip the first 501 warmup points)
            df = pd.read_csv(path).iloc[501:, FEATURE_COLS]
            if not df.empty:
                data_list.append(df.values)
        except Exception as e:
            print(f"Read failed for {f}: {e}")

    if not data_list:
        return None

    X_raw = np.vstack(data_list)
    # Standardization process
    if fit_scaler:
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw)

    return X_scaled


def main():
    # 1. Prepare file lists
    # Training set
    train_files = ["N_01.CSV", "N_02.csv", "N_03.csv"]

    # Test set: All CSV files in the directory
    all_files = os.listdir(DATA_CSV_DIR)
    test_files = [f for f in all_files if f.endswith('.csv')]
    test_files.sort()

    # Initialize scaler
    scaler = StandardScaler()

    # 2. Build training set (Raw Data)
    print(f"\n>>> [Phase 1] Loading and standardizing training data (Raw Data: {FEATURE_COLS})...")
    X_train = load_raw_data(train_files, scaler=scaler, fit_scaler=True)

    if X_train is None:
        print("Training data not obtained, exiting.")
        return

    print(f"Training set shape: {X_train.shape}")

    # 3. Define and train models
    contamination = 0.001

    classifiers = {
        'Raw-KNN': KNN(n_neighbors=5, method='largest', contamination=contamination),
        'Raw-LOF': LOF(n_neighbors=20, novelty=True, contamination=contamination),
        'Raw-GMM': GMM(n_components=3, covariance_type='full', contamination=contamination),
        'Raw-IForest': IForest(n_estimators=100, random_state=42, contamination=contamination),

        # --- Using Sklearn Native OneClassSVM (faster) ---
        # nu=0.001 corresponds to contamination, indicating a very low proportion of anomalies allowed in training set
        'Raw-OCSVM': OneClassSVM(kernel='rbf', gamma='scale', nu=0.001),

        'Raw-PCA': PCA(n_components=None, contamination=contamination),
        'Raw-COPOD': COPOD(contamination=contamination),
        'Raw-HBOS': HBOS(contamination=contamination),
        'Raw-ECOD': ECOD(contamination=contamination),
        'Raw-MCD': MCD(contamination=contamination, support_fraction=0.9),
        'Raw-ABOD': ABOD(method='fast', n_neighbors=10, contamination=contamination)

    }

    print(f"\n>>> [Phase 2] Training {len(classifiers)} classifiers (based on raw data)...")
    for name, clf in classifiers.items():
        print(f"Training {name} ...")
        clf.fit(X_train)

    # 4. Build test set and predict
    print(f"\n>>> [Phase 3] Evaluating on test set...")
    y_true_all = []
    y_scores_all = {name: [] for name in classifiers.keys()}

    for f in test_files:
        # Labeling: Starting with 'N' is 0 (Normal), otherwise 1 (Anomaly)
        label = 0 if f.upper().startswith('N') else 1

        # Load and transform test data (using the trained scaler)
        X_test = load_raw_data([f], scaler=scaler, fit_scaler=False)

        if X_test is None: continue

        # Record labels
        y_true_all.append(np.full(len(X_test), label))

        # Model anomaly score prediction
        for name, clf in classifiers.items():
            try:
                if name == 'Raw-OCSVM':
                    # Sklearn OCSVM returns distance (Positive=Normal, Negative=Anomaly)
                    # Use negative sign to convert to anomaly score (higher score = more anomalous)
                    scores = -clf.decision_function(X_test)
                else:
                    # PyOD models directly return anomaly scores
                    scores = clf.decision_function(X_test)

                y_scores_all[name].append(scores)
            except Exception as e:
                # If prediction fails, fill with 0
                y_scores_all[name].append(np.zeros(len(X_test)))

    # 5. Aggregate results and plot
    if not y_true_all:
        print("No valid test results found.")
        return

    y_true = np.concatenate(y_true_all)
    results = []

    plt.figure(figsize=(14, 12))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(classifiers))]

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

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Baseline: Raw Data Diagnosis ROC (Without LSTM)', fontsize=16)
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)

    save_path = 'results/diagnostics_plots/Raw_Data_Classic_ROC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    # 6. Print Rankings
    results.sort(key=lambda x: x[2], reverse=True)
    print("\n" + "=" * 70)
    print(f"{'Rank':<4} | {'Method':<20} | {'AP (Precision)':<10} | {'AUC (ROC)':<10}")
    print("-" * 70)
    for i, (name, auc, ap) in enumerate(results):
        print(f"{i + 1:<4} | {name:<20} | {ap:.4f}     | {auc:.4f}")
    print("=" * 70)
    print(f"\n[Done] Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
