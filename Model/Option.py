import pandas as pd
from Hydraulic_Control_Surface_Fault_Detection.Model.Visualize import save_plots
from Hydraulic_Control_Surface_Fault_Detection.Model.train import train_LSTM_models
import os
import numpy as np


"""
This file is used to train neural network models (It is better to train each model independently)
"""

if __name__ == '__main__':
    # Define the base directory for storing scalers and ensure it exists
    scaler_base_dir = os.path.join('test', 'scalers')
    # scaler_base_dir = 'scalers'
    os.makedirs(scaler_base_dir, exist_ok=True)

    # Load raw data; please change the paths according to your personal computer settings
    train_raw_data = pd.read_csv(r'F:\PythonPro\pythontorch\Hydraulic_Control_Surface_Fault_Detection\Data\data_csv\N_01.csv').values
    test_raw_data = pd.read_csv(r'F:\PythonPro\pythontorch\Hydraulic_Control_Surface_Fault_Detection\Data\data_csv\N_02.csv').values
    validata_raw_data = pd.read_csv(r'F:\PythonPro\pythontorch\Hydraulic_Control_Surface_Fault_Detection\Data\data_csv\N_03.csv').values

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

    # Define model configurations: model_test1/2/3 are component-level models, model_test_minimization/2/3 are minimal models and extensions
    models_config = {
        "model_test": {
            "type": "lstm",
            "name": "Servo Valve Velocity Prediction",
            "model_name_prefix": "model_test_56811_9",
            "input_cols": [5, 6, 8, 11],
            "output_cols": [9],
            "description": "Diagnosis of Servo Valve-related Faults",
            "lstm_params": {
                "hidden_size": 128,
                "num_layers": 1,
                "learning_rate": 0.005,
                "dropout": 0.2,
                "epoch_frequency": 1000,
                "lambda_reg": 1e-5,
                "max_norm": 3,
                "alpha": 0.38,
                "early_stop_patience": 100
            }
        },
        "model_test2": {  # Using LSTMmodel_Pressure in the train file yields better performance
            "type": "lstm",
            "name": "Servo Valve Pressure Prediction",
            "model_name_prefix": "model_test2_3910_68",
            "input_cols": [3, 9, 10],
            "output_cols": [6, 8],
            "calculate_difference": True,
            "description": "Diagnosis of Servo Valve-related Faults",
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
        "model_test3": {
            "type": "lstm",
            "name": "Control Surface Position Prediction",
            "model_name_prefix": "model_test_6811_10",
            "input_cols": [6, 8, 11],
            "output_cols": [10],
            "description": "Diagnosis of Position Sensor-related Faults",
            "lstm_params": {
                "hidden_size": 128,
                "num_layers": 1,
                "learning_rate": 0.005,
                "dropout": 0.2,
                "epoch_frequency": 1000,
                "lambda_reg": 1e-5,
                "max_norm": 3,
                "alpha": 0.42,
                "early_stop_patience": 100
            }
        },
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

    all_trained_results = {}

    for model_key, config in models_config.items():
        print(f"\n{'='*20} Training Model: {config['name']} ({config['description']}) {'='*20}")

        # Prepare data for the current model
        current_train_input = train_raw_data[501:, config['input_cols']]
        current_train_output = train_raw_data[501:, config['output_cols']]
        current_test_input = test_raw_data[501:, config['input_cols']]
        current_test_output = test_raw_data[501:, config['output_cols']]
        current_validata_input = validata_raw_data[501:, config['input_cols']]
        current_validata_output = validata_raw_data[501:, config['output_cols']]

        if config['model_name_prefix'] == "model_test_110_11":
            train_error = current_train_input[:, 0] - current_train_input[:, 1]
            test_error = current_test_input[:, 0] - current_test_input[:, 1]
            val_error = current_validata_input[:, 0] - current_validata_input[:, 1]
            current_train_input = np.column_stack((current_train_input, train_error))
            current_test_input = np.column_stack((current_test_input, test_error))
            current_validata_input = np.column_stack((current_validata_input, val_error))
        # Ensure output data is 2D (n_samples, n_features)
        if current_train_output.ndim == 1: current_train_output = current_train_output.reshape(-1, 1)
        if current_test_output.ndim == 1: current_test_output = current_test_output.reshape(-1, 1)
        if current_validata_output.ndim == 1: current_validata_output = current_validata_output.reshape(-1, 1)

        current_data = {
            "train_input": current_train_input,
            "test_input": current_test_input,
            "train_output": current_train_output,
            "test_output": current_test_output,
            "validata_input": current_validata_input,
            "validata_output": current_validata_output,
        }

        # Build parameter dictionary for current model, including dimensions and independent LSTM parameters
        model_lstm_params = config['lstm_params'].copy()
        model_lstm_params["input_size"] = current_train_input.shape[1]
        model_lstm_params["output_size"] = current_train_output.shape[1]
        current_parameter = {"LSTM": model_lstm_params}

        lstm_trainer = train_LSTM_models(
            current_data,
            current_parameter,
            model_name_prefix=config['model_name_prefix'],
            scaler_base_dir=scaler_base_dir
        )
        train_result = lstm_trainer.main()

        # # Empty training data used during testing (skip training)
        # train_result = {
        #     'train_predictions': np.array([]),  # Empty data
        #     'train_labels': np.array([]),  # Empty data
        #     'history': None,  # No history Loss
        #     'Model': None,
        #     'best_epoch': 0
        # }

        # Test the current model
        current_test_results = {
            'best_model': lstm_trainer.test(current_test_input, current_test_output, use_best_model=True),
            'final_model': lstm_trainer.test(current_test_input, current_test_output, use_best_model=False)
        }

        # Visualize results of current model and save to model-specific subdirectory
        save_plots(
            train_result=train_result,
            test_results=current_test_results,
            model=train_result['Model'],  # train_result['Model'] contains the model loaded with the best weights
            save_dir=os.path.join('test', 'results', 'visualizations', config['model_name_prefix']),
        )

        # Store training results and model objects for subsequent diagnosis
        all_trained_results[model_key] = {
            "train_result": train_result,
            "test_results": current_test_results,
            "model_obj": train_result['Model'],
            "input_scaler": lstm_trainer.input_scaler,
            "output_scaler": lstm_trainer.output_scaler,
            "model_config": config
        }

