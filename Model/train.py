from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import torch.optim
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import time
import torch.nn.functional as F
from Hydraulic_Control_Surface_Fault_Detection.Model.Enhanced_LSTM import LSTMmodel


# from Hydraulic_Control_Surface_Fault_Detection.Model.Enhanced_LSTM_Pressure import LSTMmodel  # Pressure Model


class SpectralLoss(nn.Module):
    """
    Time-domain + Frequency-domain Loss
    Considering frequency-domain loss significantly improves training results and test performance compared to using only MSE.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # Time-domain loss
        mse_loss = F.mse_loss(pred, target)
        # Frequency-domain loss
        pred_fft = torch.fft.rfft(pred, dim=1)  # Fourier transform
        target_fft = torch.fft.rfft(target, dim=1)
        # Calculate real and imaginary part losses
        freq_loss = F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(pred_fft.imag, target_fft.imag)
        return (1 - self.alpha) * mse_loss + self.alpha * freq_loss


def prepare_data(inputs, outputs, input_scaler=None, output_scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # If inputs is a 1D array, reshape it to (n_samples, 1)
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)
    # If outputs is a 1D array, reshape it to (n_samples, 1)
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    # Training mode
    if input_scaler is None:
        input_scaler = MinMaxScaler(feature_range=(-1, 1))
        inputs_normalized = input_scaler.fit_transform(inputs)
        output_scaler = MinMaxScaler(feature_range=(-1, 1))
        outputs_normalized = output_scaler.fit_transform(outputs)
    # Testing mode
    else:
        inputs_normalized = input_scaler.transform(inputs)
        outputs_normalized = output_scaler.transform(outputs)
    # Convert to tensor and add batch dimension
    inputs_tensor = torch.FloatTensor(inputs_normalized).unsqueeze(0).to(device)
    outputs_tensor = torch.FloatTensor(outputs_normalized).unsqueeze(0).to(device)
    return inputs_tensor, outputs_tensor, input_scaler, output_scaler


class train_LSTM_models:
    def __init__(self, data, parameter, model_name_prefix="default_model", scaler_base_dir="scalers"):
        self.data = data
        self.parameter = parameter
        self.model_name_prefix = model_name_prefix  # Store model prefix
        # Use time-frequency domain loss function
        self.criterion = SpectralLoss(alpha=self.parameter['LSTM']['alpha'])
        # self.criterion = nn.MSELoss()
        self.split_train_val()
        self.scaler_path = {
            'input': os.path.join(scaler_base_dir, f'{self.model_name_prefix}_input_scaler.pkl'),
            'output': os.path.join(scaler_base_dir, f'{self.model_name_prefix}_output_scaler.pkl')
        }
        # self.model_save_dir = "weights"  # Model saving directory
        self.model_save_dir = "test/weights"  # Alternative saving directory
        os.makedirs(self.model_save_dir, exist_ok=True)  # Create directory if it doesn't exist
        self.train_history = {'train_loss': [], 'val_loss': [], 'lr': []}  # Dictionary for history logging

    def split_train_val(self, test_size=0.5):
        """Do not split validation set and ensure data is 2D"""
        self.train_input = self.data['train_input']
        if self.train_input.ndim == 1:
            self.train_input = self.train_input.reshape(-1, 1)
        self.train_output = self.data['train_output']
        if self.train_output.ndim == 1:
            self.train_output = self.train_output.reshape(-1, 1)
        self.test_input = self.data['test_input']
        if self.test_input.ndim == 1:
            self.test_input = self.test_input.reshape(-1, 1)
        self.test_output = self.data['test_output']
        if self.test_output.ndim == 1:
            self.test_output = self.test_output.reshape(-1, 1)
        # Use independent validation set
        self.val_input = self.data['validata_input']
        self.val_output = self.data['validata_output']
        print(f'train_input_shape: {self.train_input.shape}')
        print(f'test_input_shape: {self.test_input.shape}')
        print(f'val_input_shape: {self.val_input.shape}')
        print("Training Data Stats - Step Amplitude Mean:", np.abs(self.train_input).mean().item())
        print("Test Data Stats - Step Amplitude Mean:", np.abs(self.test_input).mean().item())
        print("Validation Data Stats - Step Amplitude Mean:", np.abs(self.val_input).mean().item())

    def save_scalers(self, input_scaler, output_scaler):
        """Save scalers"""
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        joblib.dump(input_scaler, self.scaler_path['input'])
        joblib.dump(output_scaler, self.scaler_path['output'])
        print(f"Scalers for Model '{self.model_name_prefix}' saved to: {self.scaler_path}")

    def main(self):
        # 1. Explicitly clear CUDA cache before each model starts training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
        # Generate training data
        inputs_tensor, outputs_tensor, input_scaler, output_scaler = prepare_data(self.train_input, self.train_output)

        # Save scalers
        self.save_scalers(input_scaler, output_scaler)
        # Prepare validation data (using training scalers, i.e., current instance's input_scaler/output_scaler)
        loaded_input_scaler = joblib.load(self.scaler_path['input'])
        loaded_output_scaler = joblib.load(self.scaler_path['output'])
        val_inputs, val_outputs, _, _ = prepare_data(self.val_input, self.val_output,
                                                     input_scaler=loaded_input_scaler,
                                                     output_scaler=loaded_output_scaler)
        print("Using the full test set as the validation set")
        print(f"Training set mean: {inputs_tensor.mean():.2f}, Validation set mean: {val_inputs.mean():.2f}")
        print(f"Training set variance: {inputs_tensor.var():.2f}, Validation set variance: {val_inputs.var():.2f}")
        print("val_output shape:", val_outputs.shape)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_size = self.train_input.shape[1]
        output_size = self.train_output.shape[1]

        model = LSTMmodel(
            input_size=input_size,
            hidden_size=self.parameter['LSTM']['hidden_size'],
            num_layers=self.parameter['LSTM']['num_layers'],
            output_size=output_size,
            dropout=self.parameter['LSTM']['dropout']
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.parameter['LSTM']['learning_rate'],
            weight_decay=self.parameter['LSTM']['lambda_reg']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                               min_lr=1e-6, )  # Based on val_loss
        # scheduler = StepLR(optimizer, step_size=100, gamma=0.2)  # Based on epoch
        print('----------------Training Model (Training Data)---------------------')
        best_loss = float('inf')
        best_model_state = None
        patience = self.parameter['LSTM']['early_stop_patience']
        no_improvement_count = 0
        start_time = time.time()
        for epoch in range(self.parameter['LSTM']['epoch_frequency']):
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs_tensor)
            # Main loss
            main_loss = self.criterion(outputs, outputs_tensor)
            loss = main_loss
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.parameter['LSTM']['max_norm'])
            optimizer.step()
            current_loss = loss.item()
            self.train_history['train_loss'].append(current_loss)
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs_prediction = model(val_inputs)
                main_loss = self.criterion(val_outputs_prediction, val_outputs).item()
                val_loss = main_loss
                self.train_history['val_loss'].append(val_loss)
            # Update learning rate scheduler
            scheduler.step(val_loss)  # Scheduling based on validation loss
            current_lr = optimizer.param_groups[0]['lr']
            self.train_history['lr'].append(current_lr)
            # Print training and validation loss
            print(f'Epoch {epoch + 1}, Train Loss (Spectral): {current_loss:.6f}, Val Loss (Spectral): {val_loss:.6f}')

            # Early stopping mechanism
            if epoch == 0:  # Special handling for the first epoch
                prev_best_loss = val_loss
                best_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                best_epoch = epoch + 1
                print(f"Initial model @ epoch 1, val_loss: {best_loss:.6f}")
            else:
                if val_loss < best_loss * 0.99:
                    improvement = (prev_best_loss - val_loss) / (prev_best_loss + 1e-8) * 100
                    prev_best_loss = best_loss
                    best_loss = val_loss
                    best_model_state = deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    no_improvement_count = 0
                    print(
                        f"New best model found @ epoch {best_epoch}, val_loss: {best_loss:.6f} (Improvement: {improvement:.2f}%)")
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print(
                            f'Early stopping triggered at epoch {epoch + 1}, Best validation loss: {best_loss:.6f} (Epoch {best_epoch})')
                        break

        end_time = time.time()
        print(f'Model training duration: {end_time - start_time:.2f}s')

        final_model_state = model.state_dict().copy()  # Final epoch state
        print("\nModel state validation:")
        print(f"Best model epoch: {best_epoch}, val_loss: {best_loss:.6f}")
        print(f"Final model epoch: {epoch + 1}, val_loss: {val_loss:.6f}")
        # Check if the two model states are the same
        diff_count = 0
        for (k, v1), (_, v2) in zip(best_model_state.items(), final_model_state.items()):
            if not torch.allclose(v1, v2, rtol=1e-5, atol=1e-8):
                diff_count += 1
                if diff_count <= 3:
                    print(f"Parameter difference [{k}]: max_diff={torch.max(torch.abs(v1 - v2)):.6f}")

        print(f"Total different parameter sets: {diff_count}/{len(best_model_state)}")

        # Use final model to get training results
        model.load_state_dict(final_model_state)
        model.eval()
        with torch.no_grad():
            train_predictions = model(inputs_tensor)
            train_predictions = train_predictions.squeeze().cpu().numpy()
            train_labels = outputs_tensor.squeeze().cpu().numpy()

        # Save best model
        best_model_path = os.path.join(self.model_save_dir, f"{self.model_name_prefix}_best.pt")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'train_history': self.train_history
        }, best_model_path)

        # Save final model
        final_model_path = os.path.join(self.model_save_dir, f"{self.model_name_prefix}_last.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': final_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss,
            'train_history': self.train_history
        }, final_model_path)

        # ================= File validation start =================
        print("\nSaved file validation:")
        print(f"best.pt size: {os.path.getsize(best_model_path) / 1024:.2f} KB")
        print(f"last.pt size: {os.path.getsize(final_model_path) / 1024:.2f} KB")
        print(
            f"Modification time difference: {os.path.getmtime(final_model_path) - os.path.getmtime(best_model_path):.2f} seconds")
        return {
            'train_predictions': train_predictions,
            'train_labels': train_labels,
            'history': self.train_history,
            'best_epoch': best_epoch,
            'Model': model,
            'model_name_prefix': self.model_name_prefix  # Return prefix
        }

    def test(self, test_input, test_output, model_path=None, use_best_model=True):
        print(
            f'----------------Testing Model (Test Data) - {self.model_name_prefix}---------------------')  # Add model prefix
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load data preprocessors (using instance's self.scaler_path)
        input_scaler = joblib.load(self.scaler_path['input'])
        output_scaler = joblib.load(self.scaler_path['output'])

        # Prepare test data
        inputs_tensor, outputs_tensor, _, _ = prepare_data(
            test_input,
            test_output,
            input_scaler=input_scaler,
            output_scaler=output_scaler
        )
        true_labels = outputs_tensor.squeeze().cpu().numpy()

        if model_path is None:
            # Build default model path based on model prefix
            model_path = os.path.join("test/weights",
                                      f"{self.model_name_prefix}_best.pt" if use_best_model else f"{self.model_name_prefix}_last.pt")

        input_size_for_model = inputs_tensor.shape[2] if inputs_tensor.ndim > 1 else 1
        output_size_for_model = outputs_tensor.shape[2] if outputs_tensor.ndim > 1 else 1
        # Load saved model
        checkpoint = torch.load(model_path)
        model = LSTMmodel(
            input_size=input_size_for_model,
            hidden_size=self.parameter['LSTM']['hidden_size'],
            num_layers=self.parameter['LSTM']['num_layers'],
            output_size=output_size_for_model,
            dropout=self.parameter['LSTM']['dropout']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Perform prediction
        with torch.no_grad():
            predictions = model(inputs_tensor).squeeze().cpu().numpy()

        return predictions, true_labels

