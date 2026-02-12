import torch
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from Hydraulic_Control_Surface_Fault_Detection.Model.Enhanced_LSTM import LSTMmodel
from Hydraulic_Control_Surface_Fault_Detection.Model.train import prepare_data

"""
This file is called by various diagnostic algorithms to utilize the neural network models. 
The diagnostic heatmap-related code in this file can be ignored.
"""

plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
plt.rcParams['axes.unicode_minus'] = False  # Used to display negative signs normally
sns.set_theme(style="whitegrid")  # Use seaborn's whitegrid style

# Index 1 = Target Position
# Index 2 = Pump Speed Input
# Index 3 = Control Surface Force
# Index 4 = Pump Flow
# Index 5 = Pump Output Pressure
# Index 6 = Pressure ChamberA
# Index 7 = Servo Valve Flow
# Index 8 = Pressure ChamberB
# Index 9 = Mechanical Compliance V
# Index 10 = Mechanical Compliance X
# Index 11 = Controller Output

# Define model configurations: model_test/2/3 are component-level models, model_test_minimization/2/3 are minimal models and extensions
models_config = {
    # "model_test": {
    #     "type": "lstm",
    #     "name": "Servo Valve Velocity Prediction",
    #     "model_name_prefix": "model_test_56811_9",
    #     "input_cols": [5, 6, 8, 11],
    #     "output_cols": [9],
    #     "description": "Diagnosis of Servo Valve-related Faults",
    #     "lstm_params": {
    #         "hidden_size": 128,
    #         "num_layers": 1,
    #         "learning_rate": 0.005,
    #         "dropout": 0.2,
    #         "epoch_frequency": 1000,
    #         "lambda_reg": 1e-5,
    #         "max_norm": 3,
    #         "alpha": 0.38,
    #         "early_stop_patience": 100
    #     }
    # },
    # "model_test2": {
    #     "type": "lstm",
    #     "name": "Servo Valve Pressure Prediction",
    #     "model_name_prefix": "model_test2_3910_68",
    #     "input_cols": [3, 9, 10],
    #     "output_cols": [6, 8],
    #     "calculate_difference": True,
    #     "description": "Diagnosis of Servo Valve-related Faults",
    #     "lstm_params": {
    #         "hidden_size": 128,
    #         "num_layers": 2,
    #         "learning_rate": 0.005,
    #         "dropout": 0.2,
    #         "epoch_frequency": 1000,
    #         "lambda_reg": 1e-5,
    #         "max_norm": 3,
    #         "alpha": 0.38,
    #         "early_stop_patience": 100
    #     }
    # },
    # "model_test3": {
    #     "type": "lstm",
    #     "name": "Control Surface Position Prediction",
    #     "model_name_prefix": "model_test_6811_10",
    #     "input_cols": [6, 8, 11],
    #     "output_cols": [10],
    #     "description": "Diagnosis of Position Sensor-related Faults",
    #     "lstm_params": {
    #         "hidden_size": 128,
    #         "num_layers": 1,
    #         "learning_rate": 0.005,
    #         "dropout": 0.2,
    #         "epoch_frequency": 1000,
    #         "lambda_reg": 1e-5,
    #         "max_norm": 3,
    #         "alpha": 0.42,
    #         "early_stop_patience": 100
    #     }
    # },
    "model_test_minimization": {
        "name": "Position_Predictor",
        "input_cols": [1, 11],
        "output_cols": [10],
        "model_name_prefix": "model_test_111_10",
        "description": "minimization Model",
        "lstm_params": {
            "hidden_size": 64,
            "num_layers": 2,
            "learning_rate": 0.005,
            "dropout": 0.2,
            "epoch_frequency": 1000,
            "lambda_reg": 1e-5,
            "max_norm": 3,
            "alpha": 0.38,
            "early_stop_patience": 100
        }
    },
    "model_test_minimization2": {
        "name": "Position_Predictor",
        "input_cols": [1, 10],
        "output_cols": [11],
        "model_name_prefix": "model_test_110_11",
        "description": "minimization Model",
        "lstm_params": {
            "hidden_size": 128,
            "num_layers": 2,
            "learning_rate": 0.005,
            "dropout": 0.2,
            "epoch_frequency": 1000,
            "lambda_reg": 1e-5,
            "max_norm": 3,
            "alpha": 0.38,
            "early_stop_patience": 100
        }
    },
    "model_test_minimization3": {
        "name": "Position_Predictor",
        "input_cols": [10, 11],
        "output_cols": [1],
        "model_name_prefix": "model_test_1011_1",
        "description": "minimization Model",
        "lstm_params": {
            "hidden_size": 64,
            "num_layers": 2,
            "learning_rate": 0.005,
            "dropout": 0.2,
            "epoch_frequency": 1000,
            "lambda_reg": 1e-5,
            "max_norm": 3,
            "alpha": 0.38,
            "early_stop_patience": 100
        }
    },
}

# Path definitions, must be changed according to personal computer paths
SCALERS_DIR = r'F:\PythonPro\pythontorch\Hydraulic_Control_Surface_Fault_Detection\Model\scalers'
WEIGHTS_DIR = r'F:\PythonPro\pythontorch\Hydraulic_Control_Surface_Fault_Detection\Model\weights'
DATA_CSV_DIR = r'F:\PythonPro\pythontorch\Hydraulic_Control_Surface_Fault_Detection\Data\data_csv'


class FaultDiagnosisSystem:
    def __init__(self, models_config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.models_config = models_config
        self.device = device
        self.loaded_models = {}
        self.loaded_scalers = {}
        # Define thresholds as instance variables
        self.thresholds = {
            'model_test': 0.005,
            'model_test2': 0.05,
            'model_test3': 0.02,
            'model_math': 0.001,
            'model_test_minimization': 0.01,
            'model_test_minimization2': 0.01,
            'model_test_minimization3': 0.01,
            # 'model_test5': 0.015,
            # 'model1': 0.04,
            # 'model2': 0.02,
            # 'model3': 0.02
        }
        self._load_all_models_and_scalers()

    def _load_all_models_and_scalers(self):
        print("Loading all trained models and scalers...")
        for model_key, config in self.models_config.items():
            if config.get('type') == 'math':
                print(f"Initializing math model: {config['name']} (No weights required)")
                self.loaded_models[model_key] = "MATH_MODEL"
                continue
            prefix = config['model_name_prefix']
            print(f"Loading model file: {config['name']} ({prefix})")
            # Load scalers
            input_scaler_path = os.path.join(SCALERS_DIR, f'{prefix}_input_scaler.pkl')
            output_scaler_path = os.path.join(SCALERS_DIR, f'{prefix}_output_scaler.pkl')

            self.loaded_scalers[model_key] = {
                'input': joblib.load(input_scaler_path),
                'output': joblib.load(output_scaler_path)
            }
            print(f"Scalers loaded successfully: {input_scaler_path}, {output_scaler_path}")

            # Load model weights
            model_path = os.path.join(WEIGHTS_DIR, f'{prefix}_best.pt')
            checkpoint = torch.load(model_path, map_location=self.device)

            # Get LSTM model input/output dimensions and other parameters
            if config.get('model_name_prefix') == 'model_test_110_11':
                input_size = len(config['input_cols']) + 1
            else:
                input_size = len(config['input_cols'])

            if config.get('calculate_difference', False):
                output_size = 1
            else:
                output_size = len(config['output_cols'])
            lstm_params = config['lstm_params']
            hidden_size = lstm_params["hidden_size"]
            num_layers = lstm_params["num_layers"]
            dropout = lstm_params["dropout"]

            model = LSTMmodel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout
            ).to(self.device)

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            self.loaded_models[model_key] = model
            print(f"Model '{config['name']}' loaded successfully.")
        print("All model and scaler loading attempts completed.")
        if not self.loaded_models:
            print("Error: No models successfully loaded. Fault diagnosis cannot proceed.")
            sys.exit(1)  # Exit program if no models loaded successfully

    def get_residuals_batch(self, data_df):
        """
        Processes a batch of raw data (DataFrame), calculating predictions and residuals for each model.
        Returns a dictionary: {model_key: {'residuals': np.array, 'predictions': np.array, 'actuals': np.array}}
        All arrays are in the normalized space.
        """
        all_models_results = {}

        for model_key, config in self.models_config.items():
            if model_key not in self.loaded_models:
                continue
            if config.get('type') == 'math':
                # 1. Extract raw data directly (no normalization)
                velocity = data_df.iloc[:, config['input_cols']].values.flatten()
                actual_position = data_df.iloc[:, config['output_cols']].values.flatten()

                # 2. Get dt from configuration
                dt = config.get('dt', 0.01)

                # 3. Core math formula: Integration prediction
                # x_pred = x_initial + cumsum(v) * dt
                # np.cumsum calculates cumulative sum
                integral = np.cumsum(velocity) * dt

                # Align starting point: make prediction start equal to actual position start
                predicted_position = actual_position[0] + integral

                # 4. Calculate residuals
                # Keep dimension as (n_samples, 1) to match subsequent processing formats
                residuals = predicted_position - actual_position

                # Store into results dictionary
                all_models_results[model_key] = {
                    'residuals': residuals.reshape(-1, 1),
                    'predictions': predicted_position.reshape(-1, 1),
                    'actuals': actual_position.reshape(-1, 1)
                }
                continue  # Skip to next iteration after processing math model
            # === LSTM Model ===
            # 1. Prepare inputs
            input_data_batch = data_df.iloc[:, config['input_cols']].values  # Shape (n_samples, input_size)
            if config.get('model_name_prefix') == 'model_test_110_11':
                diff_feature = (input_data_batch[:, 0] - input_data_batch[:, 1]).reshape(-1, 1)
                input_data_batch = np.hstack([input_data_batch, diff_feature])

            if config.get('calculate_difference', False):
                raw_outputs = data_df.iloc[:, config['output_cols']].values
                actual_output_data_batch = (raw_outputs[:, 0] - raw_outputs[:, 1]).reshape(-1, 1)
            else:
                actual_output_data_batch = data_df.iloc[:, config['output_cols']].values

            # Normalize data batch
            input_tensor_batch, actual_output_tensor_batch, _, _ = prepare_data(
                input_data_batch,
                actual_output_data_batch,
                input_scaler=self.loaded_scalers[model_key]['input'],
                output_scaler=self.loaded_scalers[model_key]['output']
            )

            input_tensor_batch = input_tensor_batch.to(self.device)
            actual_output_tensor_batch = actual_output_tensor_batch.to(self.device)

            with torch.no_grad():
                predicted_output_tensor_batch = self.loaded_models[model_key](input_tensor_batch)

            predicted_output_normalized = predicted_output_tensor_batch.squeeze(0).cpu().numpy()
            actual_output_normalized = actual_output_tensor_batch.squeeze(0).cpu().numpy()

            # Calculate residuals in normalized space
            # Both operands are shape (n_samples, num_outputs), resulting in same shape for residuals
            residuals_normalized = predicted_output_normalized - actual_output_normalized

            all_models_results[model_key] = {
                'residuals': residuals_normalized,
                'predictions': predicted_output_normalized,
                'actuals': actual_output_normalized
            }
        return all_models_results

    def diagnose_and_summarize_batch(self, data_df, file_name, plot_results=True,
                                     fault_trigger_percentage_threshold=50):
        """
        Core diagnostic logic. Returns a dictionary containing the anomaly ratio for each model in the file.
        """
        print(f"Processing file {file_name}...")
        all_models_results = self.get_residuals_batch(data_df)
        plot_dir = os.path.join('results', 'diagnostics_plots', os.path.splitext(file_name)[0])
        os.makedirs(plot_dir, exist_ok=True)

        aggregated_fault_flags = {}

        file_summary_stats = {'File': file_name}

        for model_key, results in all_models_results.items():
            config = self.models_config[model_key]
            spec_config = config.get('spectral_config', {'enable': False})

            residuals = results['residuals']
            actuals = results['actuals']
            predictions = results['predictions']

            # --- Time Domain Statistics ---
            threshold = self.thresholds.get(model_key, 0.1)
            max_abs_res = np.max(np.abs(residuals), axis=1)

            # Calculate ratio of samples exceeding threshold (0.0 - 1.0)
            faulty_rows_ratio = np.sum(max_abs_res > threshold) / len(residuals)
            faulty_rows_pct = faulty_rows_ratio * 100

            # Store ratio in summary dictionary (for heatmap generation)
            file_summary_stats[model_key] = faulty_rows_ratio

            is_time_fault = faulty_rows_pct > fault_trigger_percentage_threshold

            print(f"[{config['name']}] Status Report:")
            print(f"> Time Domain Anomaly Ratio: {faulty_rows_pct:.2f}% (Threshold: {threshold})")

            aggregated_fault_flags[model_key] = is_time_fault

            if plot_results:
                self.plot_residuals(model_key, results, file_name, plot_dir)
        final_msg = self.interpret_fault_flags(aggregated_fault_flags)
        print(f"Overall diagnosis for file {file_name}: {final_msg}")

        # Return statistical data
        return file_summary_stats

    def interpret_fault_flags(self, fault_flags):
        """
        Lists which models detected abnormal data.
        """
        active_models = [self.models_config[k]['name'] for k, v in fault_flags.items() if v]

        if not active_models:
            return "Normal (all model metrics are within thresholds)."

        return f"Anomaly detected. Triggered models: {', '.join(active_models)}"

    def plot_residuals(self, model_key, results_dict, file_name, save_dir):
        model_name = self.models_config[model_key]['name']
        output_cols = self.models_config[model_key]['output_cols']

        # Calculate differential pressure mode
        if self.models_config[model_key].get('calculate_difference', False):
            display_output_cols = [f"Diff({output_cols[0]}-{output_cols[1]})"]
        else:
            display_output_cols = output_cols

        residuals = results_dict['residuals']
        predictions = results_dict['predictions']
        actuals = results_dict['actuals']
        threshold = self.thresholds.get(model_key, 0.1)

        num_outputs = residuals.shape[1]
        time_steps = np.arange(len(residuals))
        fig_height = max(5 * num_outputs * 2, 8)
        fig, axes = plt.subplots(num_outputs * 2, 1, figsize=(15, fig_height), sharex=True)

        if num_outputs == 1:
            axes = np.array([axes]).flatten()

        for i in range(num_outputs):
            col_label = display_output_cols[i] if i < len(display_output_cols) else f"Output {i}"

            # Actual vs Predicted
            ax_pred = axes[i * 2]
            sns.lineplot(x=time_steps, y=actuals[:, i].flatten(), label='Actual (Norm)', ax=ax_pred, alpha=0.7)
            sns.lineplot(x=time_steps, y=predictions[:, i].flatten(), label='Predicted (Norm)', ax=ax_pred, alpha=0.7,
                         linestyle='--')
            ax_pred.set_title(f'{model_name} - {col_label} - Actual vs. Predicted')
            ax_pred.set_ylabel('Normalized Value')
            ax_pred.legend()
            ax_pred.grid(True)

            # Residuals
            ax_res = axes[i * 2 + 1]
            sns.lineplot(x=time_steps, y=residuals[:, i].flatten(), label='Residuals', color='red', ax=ax_res,
                         alpha=0.7)
            ax_res.axhline(threshold, color='green', linestyle='--')
            ax_res.axhline(-threshold, color='green', linestyle='--')
            ax_res.set_title(f'{model_name} - {col_label} - Residuals')
            ax_res.set_ylabel('Normalized Residual')
            ax_res.fill_between(time_steps, residuals[:, i], threshold, where=residuals[:, i] > threshold,
                                color='orange', alpha=0.3)
            ax_res.fill_between(time_steps, residuals[:, i], -threshold, where=residuals[:, i] < -threshold,
                                color='orange', alpha=0.3)
            ax_res.grid(True)

        plt.xlabel('Time Step')
        plt.tight_layout()
        plot_save_path = os.path.join(save_dir, f'{os.path.splitext(file_name)[0]}_{model_key}_residuals.png')
        plt.savefig(plot_save_path)
        plt.close(fig)

    def plot_diagnosis_summary_heatmap(self, summary_list):
        if not summary_list:
            print("No diagnostic data collected, skipping plotting.")
            return

        df = pd.DataFrame(summary_list)
        if 'File' in df.columns:
            df.set_index('File', inplace=True)

        # Sort by filename
        df.sort_index(inplace=True)

        # Plotting
        plt.figure(figsize=(10, max(6, len(df) * 0.5 + 2)))

        # Color mapping: 0% (White) -> 20% (Red)
        # Any anomaly ratio exceeding 20% will display as the deepest red
        ax = sns.heatmap(df, annot=True, fmt=".1%", cmap="Reds",
                         vmin=0, vmax=0.2, linewidths=1, linecolor='white')

        plt.title("LSTM Diagnosis Summary: Anomaly Percentage by Model\n(Time Domain Residual > Threshold)",
                  fontsize=14)
        plt.ylabel("Test Files")
        plt.xlabel("Models")

        plt.tight_layout()
        save_path = os.path.join('results', 'diagnostics_plots', 'LSTM_Summary_Heatmap.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"\n[Summary] LSTM diagnostic summary heatmap saved: {save_path}")


if __name__ == '__main__':
    diagnosis_system = FaultDiagnosisSystem(models_config)
    print("\n--- Starting Fault Diagnosis Simulation ---")
    # Collect diagnostic results for all files
    all_summaries = []
    # --- Process normal operation data ---
    normal_files = [f for f in os.listdir(DATA_CSV_DIR) if f.startswith('N_') and f.endswith('.csv')]
    normal_files.sort()
    for f in normal_files:
        path = os.path.join(DATA_CSV_DIR, f)
        print(f"\n{'=' * 10} Diagnosing normal operation data: {f} {'=' * 10}")
        df = pd.read_csv(path).iloc[501:].copy()
        # Call diagnosis and collect return value
        summary = diagnosis_system.diagnose_and_summarize_batch(df, f, plot_results=True,
                                                                fault_trigger_percentage_threshold=8)
        all_summaries.append(summary)

    # --- Process fault data files ---
    fault_files = [f for f in os.listdir(DATA_CSV_DIR) if f.startswith('F_') and f.endswith('.csv')]
    fault_files.sort()  # Sort to process in order like F_01_A, F_01_B, etc.

    for fault_file_name in fault_files:
        fault_file_path = os.path.join(DATA_CSV_DIR, fault_file_name)
        print(f"\n{'=' * 10} Diagnosing fault data: {fault_file_name} {'=' * 10}")
        # Read data starting from line 502
        fault_df = pd.read_csv(fault_file_path).iloc[501:].copy()
        # Use higher threshold for fault data
        summary = diagnosis_system.diagnose_and_summarize_batch(fault_df, fault_file_name, plot_results=True,
                                                                fault_trigger_percentage_threshold=15)
        all_summaries.append(summary)

    # --- Finally: Generate panorama heatmap ---
    print("\n>>> Generating LSTM diagnosis panorama heatmap...")
    diagnosis_system.plot_diagnosis_summary_heatmap(all_summaries)
    print(">>> Diagnosis completed.")

