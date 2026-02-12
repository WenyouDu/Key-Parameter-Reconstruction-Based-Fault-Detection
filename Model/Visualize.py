import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime
import numpy as np
from scipy import signal

# ==================== Global Font Settings ====================
plt.rcParams.update({
    'font.family': 'serif',  # Prioritize serif fonts
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 16,  # Basic font size
    'axes.titlesize': 20,  # Subplot title font size
    'axes.labelsize': 16,  # Axis label font size
    'xtick.labelsize': 14,  # x-axis tick label font size
    'ytick.labelsize': 14,  # y-axis tick label font size
    'legend.fontsize': 14,  # Legend font size adjusted to 14
    'figure.titlesize': 30  # Figure suptitle font size
})

plt.rcParams['axes.unicode_minus'] = False


def save_plots(train_result, test_results, model, save_dir='results/visualizations'):
    """
    Train and test LSTM model and save prediction comparison plots
    (Includes detailed logs, supports multi-output R2, RMSE, and spectral error display)
    Args:
        train_result (dict): Training set results dictionary.
        test_results (dict): Testing set results dictionary.
        model: Trained model object.
        save_dir (str): Directory to save plots.
    """
    # Create log header with timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_str = f"\n\n{'=' * 50}\nTraining Log {current_time}\n"

    # Create plot saving directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        log_str += f"[System] Created plot directory: {save_dir}\n"
    else:
        log_str += f"[System] Using existing directory: {save_dir}\n"

    # Extract data from input parameters
    Y_train = train_result['train_predictions']
    Y_train_true = train_result['train_labels']
    train_history = train_result['history']
    best_epoch = train_result['best_epoch']

    # Force all relevant arrays to be at least 2D (n_samples, n_features)
    if Y_train.ndim == 1:
        Y_train = Y_train.reshape(-1, 1)
    if Y_train_true.ndim == 1:
        Y_train_true = Y_train_true.reshape(-1, 1)

    # Get number of output dimensions
    n_outputs = Y_train.shape[1]
    n_outputs_test = n_outputs  # Assume test set output dimensions match training set

    # ==================== Unified Height Setting (Used for all time and frequency plots) ====================
    unified_plot_height_factor = 4.0

    # ================= Phase 1: Training Results Visualization and Metrics (Time Domain) =================
    log_str += "\n[Phase 1] Training Results Visualization and Metrics Calculation\n"
    log_str += f"- Number of Output Dimensions: {n_outputs}\n"

    try:
        fig_train_time, axes_train_time = plt.subplots(n_outputs, 1,
                                                       figsize=(14, unified_plot_height_factor * n_outputs), dpi=300)
        if n_outputs == 1:
            axes_train_time = [axes_train_time]  # Ensure axes is iterable

        train_r2_per_output = []
        train_rmse_per_output = []

        # --- Local Zoom Configuration ---
        zoom_output_channel = 0
        fixed_zoom_point = 5000
        zoom_range_x = 100
        # ------------------------

        for i in range(n_outputs):
            ax = axes_train_time[i]
            true_values = Y_train_true[:, i]
            predicted_values = Y_train[:, i]

            # Calculate R2 and RMSE for current output channel
            r2 = r2_score(true_values, predicted_values)
            train_r2_per_output.append(r2)
            rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
            train_rmse_per_output.append(rmse)

            # Plot with adjusted linewidth and alpha
            ax.plot(true_values, label='True', color='blue', alpha=0.8, linewidth=1.8)
            ax.plot(predicted_values, label='Predicted', color='red', alpha=0.8, linewidth=1.5, linestyle='--')
            ax.set_title(f'Output {i + 1} - Training (R² = {r2:.5f})', fontsize=plt.rcParams['axes.titlesize'])
            ax.set_xlabel('Time Steps', fontsize=plt.rcParams['axes.labelsize'])
            ax.set_ylabel('Value', fontsize=plt.rcParams['axes.labelsize'])
            ax.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'])
            ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
            ax.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])
            ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.05)

            # --- Add Inset Zoom (only for specified channel) ---
            if i == zoom_output_channel:
                # Define zoom area
                x_start = max(0, fixed_zoom_point - zoom_range_x)
                x_end = min(len(true_values), fixed_zoom_point + zoom_range_x)
                # Calculate Y-axis range based on data within zoom area
                y_min_data_in_zoom = np.min(true_values[x_start:x_end])
                y_max_data_in_zoom = np.max(true_values[x_start:x_end])

                # Default minimum height if range is too small
                if y_max_data_in_zoom - y_min_data_in_zoom < 0.05:
                    y_center = (y_min_data_in_zoom + y_max_data_in_zoom) / 2
                    y_min_data_in_zoom = y_center - 0.025
                    y_max_data_in_zoom = y_center + 0.025

                # ============== Adjust Y-axis margin ==============
                y_margin_factor_zoom = 0.2
                y_range_in_zoom = y_max_data_in_zoom - y_min_data_in_zoom

                y_min_zoom = y_min_data_in_zoom - y_range_in_zoom * y_margin_factor_zoom
                y_max_zoom = y_max_data_in_zoom + y_range_in_zoom * y_margin_factor_zoom

                # Ensure reasonable Y range (0-1.1 for normalized data)
                y_min_zoom = max(0, y_min_zoom)
                y_max_zoom = min(1.1, y_max_zoom)
                # =======================================================

                # Create inset axes [left, bottom, width, height] as fraction of main axes
                inset_ax = ax.inset_axes([0.03, 0.7, 0.3, 0.3])
                x_values_zoom = np.arange(x_start, x_end)
                inset_ax.plot(x_values_zoom, true_values[x_start:x_end], label='True', color='blue', alpha=0.9,
                              linewidth=1.8)
                inset_ax.plot(x_values_zoom, predicted_values[x_start:x_end], label='Predicted', color='red', alpha=0.9,
                              linewidth=1.5, linestyle='--')

                inset_ax.set_xlim(x_start, x_end)
                inset_ax.set_ylim(y_min_zoom, y_max_zoom)

                inset_ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'] - 2)
                inset_ax.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'] - 2)
                inset_ax.grid(True, linestyle=':', alpha=0.7, linewidth=0.05)

                # Draw zoom indication box/lines
                ax.indicate_inset_zoom(inset_ax, edgecolor="black", linewidth=2.0, linestyle='-')
            # --- End Local Zoom ---

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        overall_train_r2 = r2_score(Y_train_true, Y_train)
        overall_train_rmse = np.sqrt(np.mean((Y_train_true - Y_train) ** 2))

        fig_train_time.suptitle(f'Training Predictions (Overall R² = {overall_train_r2:.5f})',
                                fontsize=plt.rcParams['figure.titlesize'])
        train_plot_path = os.path.join(save_dir, 'training_predictions.png')
        plt.savefig(train_plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig_train_time)

        log_str += f"- Training Set Overall R2 Score: {overall_train_r2:.6f}\n"
        log_str += f"- Training Set Overall RMSE: {overall_train_rmse:.6f}\n"
        for i, r2 in enumerate(train_r2_per_output):
            log_str += f"- Training Output {i + 1} R2 Score: {r2:.6f}\n"
        for i, rmse in enumerate(train_rmse_per_output):
            log_str += f"- Training Output {i + 1} RMSE: {rmse:.6f}\n"

        log_str += f"- Training result plot saved to: {train_plot_path}\n"
        print(
            f"[Train Finished] Overall R2: {overall_train_r2:.5f} | Overall RMSE: {overall_train_rmse:.5f} | Per-output R2: {[f'{r:.5f}' for r in train_r2_per_output]} | Per-output RMSE: {[f'{r:.5f}' for r in train_rmse_per_output]} | Plots saved")

    except Exception as e:
        log_str += f"[Error] Training visualization failed: {str(e)}\n"
        print(f"Training visualization error: {e}")

    # ================= Phase 2: Testing Results Visualization and Metrics (Time Domain) =================
    log_str += "\n[Phase 2] Testing Results Visualization and Metrics Calculation\n"
    try:
        best_pred, best_true = test_results['best_model']
        final_pred, final_true = test_results['final_model']

        if best_pred.ndim == 1:
            best_pred = best_pred.reshape(-1, 1)
        if best_true.ndim == 1:
            best_true = best_true.reshape(-1, 1)
        if final_pred.ndim == 1:
            final_pred = final_pred.reshape(-1, 1)
        if final_true.ndim == 1:
            final_true = final_true.reshape(-1, 1)

        best_r2_per_output = []
        best_rmse_per_output = []
        final_r2_per_output = []
        final_rmse_per_output = []

        # Plot Best Model Test Results
        fig_test_best_time, axes_test_best_time = plt.subplots(n_outputs_test, 1, figsize=(
            14, unified_plot_height_factor * n_outputs_test), dpi=300)
        if n_outputs_test == 1:
            axes_test_best_time = [axes_test_best_time]

        for i in range(n_outputs_test):
            ax = axes_test_best_time[i]
            true_values = best_true[:, i]
            predicted_values = best_pred[:, i]
            r2 = r2_score(true_values, predicted_values)
            best_r2_per_output.append(r2)
            rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
            best_rmse_per_output.append(rmse)

            ax.plot(true_values, alpha=0.8, linewidth=1.8, label='True', color='blue')
            ax.plot(predicted_values, alpha=0.8, linewidth=1.5, label='Predicted', color='red', linestyle='--')
            ax.set_title(f'Output {i + 1} - Best Model Testing (R² = {r2:.5f})',
                         fontsize=plt.rcParams['axes.titlesize'])
            ax.set_xlabel('Time Steps', fontsize=plt.rcParams['axes.labelsize'])
            ax.set_ylabel('Value', fontsize=plt.rcParams['axes.labelsize'])
            ax.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'])
            ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
            ax.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])
            ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.05)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        overall_best_r2 = r2_score(best_true, best_pred)
        overall_best_rmse = np.sqrt(np.mean((best_true - best_pred) ** 2))

        fig_test_best_time.suptitle(f'Testing Predictions (Best Model, Overall R² = {overall_best_r2:.5f})',
                                    fontsize=plt.rcParams['figure.titlesize'])

        best_path = os.path.join(save_dir, 'best_model_comparison.png')
        plt.savefig(best_path, bbox_inches='tight', dpi=300)
        plt.close(fig_test_best_time)

        # Plot Final Model Test Results
        fig_test_final_time, axes_test_final_time = plt.subplots(n_outputs_test, 1, figsize=(
            14, unified_plot_height_factor * n_outputs_test), dpi=300)
        if n_outputs_test == 1:
            axes_test_final_time = [axes_test_final_time]

        for i in range(n_outputs_test):
            ax = axes_test_final_time[i]
            true_values = final_true[:, i]
            predicted_values = final_pred[:, i]
            r2 = r2_score(true_values, predicted_values)
            final_r2_per_output.append(r2)
            rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
            final_rmse_per_output.append(rmse)

            ax.plot(true_values, alpha=0.8, linewidth=1.8, label='True', color='blue')
            ax.plot(predicted_values, alpha=0.8, linewidth=1.5, label='Predicted', color='red', linestyle='--')
            ax.set_title(f'Output {i + 1} - Final Model Testing (R² = {r2:.5f})',
                         fontsize=plt.rcParams['axes.titlesize'])
            ax.set_xlabel('Time Steps', fontsize=plt.rcParams['axes.labelsize'])
            ax.set_ylabel('Value', fontsize=plt.rcParams['axes.labelsize'])
            ax.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'])
            ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
            ax.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])
            ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.05)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        overall_final_r2 = r2_score(final_true, final_pred)
        overall_final_rmse = np.sqrt(np.mean((final_true - final_pred) ** 2))

        fig_test_final_time.suptitle(f'Testing Predictions (Final Model, Overall R² = {overall_final_r2:.5f})',
                                     fontsize=plt.rcParams['figure.titlesize'])

        final_path = os.path.join(save_dir, 'final_model_comparison.png')
        plt.savefig(final_path, bbox_inches='tight', dpi=300)
        plt.close(fig_test_final_time)

        log_str += f"- Best Model Test Overall R²: {overall_best_r2:.6f}\n"
        log_str += f"- Best Model Test Overall RMSE: {overall_best_rmse:.6f}\n"
        for i, r2 in enumerate(best_r2_per_output):
            log_str += f"- Best Model Test Output {i + 1} R²: {r2:.6f}\n"
        for i, rmse in enumerate(best_rmse_per_output):
            log_str += f"- Best Model Test Output {i + 1} RMSE: {rmse:.6f}\n"

        log_str += f"- Final Model Test Overall R²: {overall_final_r2:.6f}\n"
        log_str += f"- Final Model Test Overall RMSE: {overall_final_rmse:.6f}\n"
        for i, r2 in enumerate(final_r2_per_output):
            log_str += f"- Final Model Test Output {i + 1} R²: {r2:.6f}\n"
        for i, rmse in enumerate(final_rmse_per_output):
            log_str += f"- Final Model Test Output {i + 1} RMSE: {rmse:.6f}\n"

        print(
            f"[Test Finished] Best Model Overall R2: {overall_best_r2:.5f} | RMSE: {overall_best_rmse:.5f} | Best Test R2 (per output): {[f'{r:.5f}' for r in best_r2_per_output]} | Plots saved")
        print(
            f"Final Model Overall R2: {overall_final_r2:.5f} | RMSE: {overall_final_rmse:.5f} | Final Test R2 (per output): {[f'{r:.5f}' for r in final_r2_per_output]} | Plots saved")
        print(f"Model comparison plots saved to: {save_dir}")
    except Exception as e:
        log_str += f"[Error] Test result visualization failed: {str(e)}\n"
        print(f"Test result visualization error: {e}")

    # ================= Phase 3: Training Process Logging (Loss Curves) =================
    log_str += "\n[Phase 3] Training Process Logging\n"
    try:
        if train_history:
            fig_loss = plt.figure(figsize=(13, 5), dpi=300)
            if 'train_loss' in train_history:
                loss_data = train_history['train_loss']
            elif 'loss' in train_history:
                loss_data = train_history['loss']
            else:
                loss_data = train_history.history['loss']

            plt.plot(loss_data, label='Train Loss', linewidth=1.8)

            if 'val_loss' in train_history:
                val_loss_data = train_history['val_loss']
                plt.plot(val_loss_data, label='Validation Loss', linewidth=1.8)
            elif hasattr(train_history, 'history') and 'val_loss' in train_history.history:
                plt.plot(train_history.history['val_loss'], label='Validation Loss', linewidth=1.8)

            plt.title('Model Loss Over Epochs', fontsize=plt.rcParams['axes.titlesize'])
            plt.xlabel('Epoch', fontsize=plt.rcParams['axes.labelsize'])
            plt.ylabel('Loss', fontsize=plt.rcParams['axes.labelsize'])
            plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})', linewidth=1)
            plt.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'])
            plt.grid(True, linestyle=':', alpha=0.6, linewidth=0.05)
            plt.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
            plt.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])

            loss_plot_path = os.path.join(save_dir, 'model_loss.png')
            plt.savefig(loss_plot_path, bbox_inches='tight', dpi=300)
            plt.close(fig_loss)
            log_str += f"- Model loss curve plot saved to: {loss_plot_path}\n"

            log_str += "- Training Loss History:\n"
            for epoch, loss_val in enumerate(loss_data):
                val_loss_info = ""
                if 'val_loss' in train_history:
                    val_loss_info = f", Val Loss: {train_history['val_loss'][epoch]:.6f}"
                elif hasattr(train_history, 'history') and 'val_loss' in train_history.history:
                    val_loss_info = f", Val Loss: {train_history.history['val_loss'][epoch]:.6f}"
                log_str += f"  Epoch {epoch + 1}: Train Loss: {loss_val:.6f}{val_loss_info}\n"
            log_str += f"- Best Training Epoch: {best_epoch}\n"
        else:
            log_str += "[Warning] No training history provided, skipping loss curve and logs.\n"
    except Exception as e:
        log_str += f"[Error] Training process logging failed: {str(e)}\n"
        print(f"Training process logging error: {e}")

    # ================= Phase 4: Training Data Spectral Analysis (Combined Plots) =================
    log_str += "\n[Phase 4] Training Data Spectral Analysis\n"
    train_spectral_errors_per_output = []
    train_overall_spectral_error_sum = []

    try:
        fixed_error_ylim = (-13, 18)
        fig_train_freq, axes_train_freq = plt.subplots(n_outputs, 2,
                                                       figsize=(16, unified_plot_height_factor * n_outputs),
                                                       sharex=True, dpi=300)
        if n_outputs == 1:
            axes_train_freq = np.array([axes_train_freq])

        eval_freq_range_train = (1, 100)

        for i in range(n_outputs):
            ax1 = axes_train_freq[i, 0]
            ax2 = axes_train_freq[i, 1]

            fixed_window = slice(0, min(8000, len(Y_train_true)))
            y_true = Y_train_true[fixed_window, i]
            y_pred = Y_train[fixed_window, i]
            n_current = min(len(y_true), len(y_pred))
            residual = y_true - y_pred

            # ----------------- Power Spectral Density Analysis (dB) -----------------
            fs = 1000
            nperseg = min(512, n_current // 4)
            f_true, Pxx_true = signal.welch(y_true, fs=fs, nperseg=nperseg)
            f_pred, Pxx_pred = signal.welch(y_pred, fs=fs, nperseg=nperseg)

            Pxx_true_db = 10 * np.log10(Pxx_true + 1e-20)
            Pxx_pred_db = 10 * np.log10(Pxx_pred + 1e-20)
            hf_band_attenuation_mask = (f_true > 100) & (f_true < fs / 2)
            hf_true_mean_db = np.nanmean(Pxx_true_db[hf_band_attenuation_mask]) if np.any(
                hf_band_attenuation_mask) else np.nan
            hf_pred_mean_db = np.nanmean(Pxx_pred_db[hf_band_attenuation_mask]) if np.any(
                hf_band_attenuation_mask) else np.nan
            hf_attenuation_db = hf_pred_mean_db - hf_true_mean_db if not (
                    np.isnan(hf_true_mean_db) or np.isnan(hf_pred_mean_db)) else np.nan

            ax1.plot(f_true, Pxx_true_db, 'b-', label=f'True PSD (HF mean={hf_true_mean_db:.1f} dB)',
                     linewidth=1.8)
            ax1.plot(f_pred, Pxx_pred_db, 'r', label=f'Pred PSD (HF attenuation={hf_attenuation_db:.1f} dB)',
                     linestyle='--', linewidth=1.5)
            ax1.axvline(100, color='gray', linestyle='--', alpha=0.7, label='HF Band Start (100Hz)', linewidth=0.8)
            ax1.set_title(f'Output {i + 1} - Training Power Spectral Density', fontsize=plt.rcParams['axes.titlesize'])
            ax1.set_xlabel('Frequency [Hz]', fontsize=plt.rcParams['axes.labelsize'])
            ax1.set_ylabel('PSD [dB]', fontsize=plt.rcParams['axes.labelsize'])
            ax1.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'])
            ax1.grid(True, linestyle=':', alpha=0.6, linewidth=0.05)
            ax1.set_xlim(0, fs / 2)
            ax1.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
            ax1.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])

            # ----------------- Spectral Error Analysis -----------------
            freqs = np.fft.rfftfreq(n_current, d=1 / fs)
            fft_true = np.abs(np.fft.rfft(y_true))
            fft_pred = np.abs(np.fft.rfft(y_pred))
            spectral_error = 20 * np.log10(np.clip(fft_pred / (np.clip(fft_true, 1e-12, None)), 1e-2, 1e2))

            eval_freq_mask = (freqs >= eval_freq_range_train[0]) & (freqs <= eval_freq_range_train[1])
            mean_abs_spectral_error_eval_range = np.nanmean(np.abs(spectral_error[eval_freq_mask])) if np.any(
                eval_freq_mask) else np.nan
            train_spectral_errors_per_output.append(mean_abs_spectral_error_eval_range)
            train_overall_spectral_error_sum.extend(np.abs(spectral_error[eval_freq_mask]))

            ax2.plot(freqs, spectral_error, 'b-', linewidth=1.8)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='0 dB reference line')
            ax2.axhline(3, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
            ax2.axhline(-3, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label=r'$\pm3$ dB tolerance')

            ax2.set_ylim(fixed_error_ylim)
            ax2.set_xlim(0, fs / 2)

            display_freq_mask = (freqs > 0) & (freqs < fs / 2)
            clipped_spectral_error = np.clip(spectral_error, fixed_error_ylim[0], fixed_error_ylim[1])

            if np.any(clipped_spectral_error > 3):
                ax2.fill_between(freqs, 3, fixed_error_ylim[1],
                                 where=(clipped_spectral_error > 3) & display_freq_mask,
                                 color='orange', alpha=0.2, label=f'Error > +3dB')
            if np.any(clipped_spectral_error < -3):
                ax2.fill_between(freqs, fixed_error_ylim[0], -3,
                                 where=(clipped_spectral_error < -3) & display_freq_mask,
                                 color='lightblue', alpha=0.2, label=f'Error < -3dB')

            ax2.set_title(
                f'Output {i + 1} - Training Spectral Error (Avg Abs={mean_abs_spectral_error_eval_range:.1f} dB)',
                fontsize=plt.rcParams['axes.titlesize'])
            ax2.set_xlabel('Frequency [Hz]', fontsize=plt.rcParams['axes.labelsize'])
            ax2.set_ylabel('Amplitude Error (dB)', fontsize=plt.rcParams['axes.labelsize'])
            ax2.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'])
            ax2.grid(True, linestyle=':', alpha=0.6, linewidth=0.05)
            ax2.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
            ax2.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])

            log_str += (f"- Training Output {i + 1} Spectral Analysis:\n"
                        f"  • RMSE: {np.sqrt(np.mean(residual ** 2)):.6f}\n"
                        f"  • Mean Abs Spectral Error (1-{eval_freq_range_train[1]}Hz): {mean_abs_spectral_error_eval_range:.6f} dB\n"
                        f"  • PSD High-freq Attenuation(>100Hz): {hf_attenuation_db:.1f} dB (Pred - True)\n"
                        f"  • Peak Frequency Offset: {f_true[np.argmax(Pxx_true)] - f_pred[np.argmax(Pxx_pred)]:.2f} Hz\n")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        overall_train_avg_abs_spectral_error = np.nanmean(train_overall_spectral_error_sum) if len(
            train_overall_spectral_error_sum) > 0 else np.nan

        fig_train_freq.suptitle(
            f"Training Spectral Analysis - Overall Avg Abs Spectral Error = {overall_train_avg_abs_spectral_error:.1f} dB",
            fontsize=plt.rcParams['figure.titlesize'])

        combined_spectral_path_train = os.path.join(save_dir, 'spectral_analysis_all_train_channels.png')
        plt.savefig(combined_spectral_path_train, bbox_inches='tight', dpi=300)
        plt.close(fig_train_freq)

        print(f"Aggregated training spectral analysis plot saved to: {combined_spectral_path_train}")
        print(f"Training Frequency Domain Error (1-{eval_freq_range_train[1]}Hz):")
        for i, mae in enumerate(train_spectral_errors_per_output):
            log_str += f"- Training Output {i + 1} Mean Abs Spectral Error (1-{eval_freq_range_train[1]}Hz): {mae:.6f} dB\n"
            print(f"  - Output {i + 1} Mean Abs Spectral Error: {mae:.6f} dB")
        log_str += f"- Training Set Overall Mean Abs Spectral Error (1-{eval_freq_range_train[1]}Hz): {overall_train_avg_abs_spectral_error:.6f} dB\n"
        print(f"Training Set Overall Mean Abs Spectral Error: {overall_train_avg_abs_spectral_error:.6f} dB")
    except Exception as e:
        log_str += f"[Warning] Training spectral analysis failed: {str(e)}\n"
        print(f"Training spectral analysis error: {e}")

    # ============== Test Set Spectral Analysis (Combined Plots) ===============
    log_str += "\n[Phase 5] Test Data Spectral Analysis\n"
    test_spectral_errors_per_output = []
    test_overall_spectral_error_sum = []

    try:
        best_pred, best_true = test_results['best_model']

        if best_pred.ndim == 1:
            best_pred = best_pred.reshape(-1, 1)
        if best_true.ndim == 1:
            best_true = best_true.reshape(-1, 1)

        fixed_error_ylim = (-13, 18)
        fig_test_freq, axes_test_freq = plt.subplots(n_outputs_test, 2,
                                                     figsize=(16, unified_plot_height_factor * n_outputs_test),
                                                     sharex=True, dpi=300)

        if n_outputs_test == 1:
            axes_test_freq = np.array([axes_test_freq])

        eval_freq_range_test = (1, 100)

        for i in range(n_outputs_test):
            ax1 = axes_test_freq[i, 0]
            ax2 = axes_test_freq[i, 1]

            fixed_window = slice(0, min(8000, len(best_true)))
            y_true = best_true[fixed_window, i]
            y_pred = best_pred[fixed_window, i]
            n_current = min(len(y_true), len(y_pred))
            residual = y_true - y_pred

            # ----------------- Power Spectral Density Analysis (dB) -----------------
            fs = 1000
            nperseg = min(512, n_current // 4)
            f_true, Pxx_true = signal.welch(y_true, fs=fs, nperseg=nperseg)
            f_pred, Pxx_pred = signal.welch(y_pred, fs=fs, nperseg=nperseg)

            Pxx_true_db = 10 * np.log10(Pxx_true + 1e-20)
            Pxx_pred_db = 10 * np.log10(Pxx_pred + 1e-20)
            hf_band_attenuation_mask = (f_true > 100) & (f_true < fs / 2)
            hf_true_mean_db = np.nanmean(Pxx_true_db[hf_band_attenuation_mask]) if np.any(
                hf_band_attenuation_mask) else np.nan
            hf_pred_mean_db = np.nanmean(Pxx_pred_db[hf_band_attenuation_mask]) if np.any(
                hf_band_attenuation_mask) else np.nan
            hf_attenuation_db = hf_pred_mean_db - hf_true_mean_db if not (
                    np.isnan(hf_true_mean_db) or np.isnan(hf_pred_mean_db)) else np.nan

            ax1.plot(f_true, Pxx_true_db, 'b-', label=f'True PSD (HF mean={hf_true_mean_db:.1f} dB)',
                     linewidth=1.8)
            ax1.plot(f_pred, Pxx_pred_db, 'r', label=f'Pred PSD (HF attenuation={hf_attenuation_db:.1f} dB)',
                     linestyle='--', linewidth=1.5)
            ax1.axvline(100, color='gray', linestyle='--', alpha=0.7, label='HF Band Start (100Hz)', linewidth=0.8)
            ax1.set_title(f'Output {i + 1} - Power Spectral Density', fontsize=plt.rcParams['axes.titlesize'])
            ax1.set_xlabel('Frequency [Hz]', fontsize=plt.rcParams['axes.labelsize'])
            ax1.set_ylabel('PSD [dB]', fontsize=plt.rcParams['axes.labelsize'])
            ax1.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'])
            ax1.grid(True, linestyle=':', alpha=0.6, linewidth=0.05)
            ax1.set_xlim(0, fs / 2)
            ax1.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
            ax1.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])

            # ----------------- Spectral Error Analysis -----------------
            freqs = np.fft.rfftfreq(n_current, d=1 / fs)
            fft_true = np.abs(np.fft.rfft(y_true))
            fft_pred = np.abs(np.fft.rfft(y_pred))
            spectral_error = 20 * np.log10(np.clip(fft_pred / (np.clip(fft_true, 1e-12, None)), 1e-2, 1e2))

            eval_freq_mask = (freqs >= eval_freq_range_test[0]) & (freqs <= eval_freq_range_test[1])
            mean_abs_spectral_error_eval_range = np.nanmean(np.abs(spectral_error[eval_freq_mask])) if np.any(
                eval_freq_mask) else np.nan
            test_spectral_errors_per_output.append(mean_abs_spectral_error_eval_range)
            test_overall_spectral_error_sum.extend(np.abs(spectral_error[eval_freq_mask]))

            ax2.plot(freqs, spectral_error, 'b-', linewidth=1.8)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='0 dB reference line')
            ax2.axhline(3, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
            ax2.axhline(-3, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label=r'$\pm3$ dB tolerance')

            ax2.set_ylim(fixed_error_ylim)
            ax2.set_xlim(0, fs / 2)

            display_freq_mask = (freqs > 0) & (freqs < fs / 2)
            clipped_spectral_error = np.clip(spectral_error, fixed_error_ylim[0], fixed_error_ylim[1])

            if np.any(clipped_spectral_error > 3):
                ax2.fill_between(freqs, 3, fixed_error_ylim[1],
                                 where=(clipped_spectral_error > 3) & display_freq_mask,
                                 color='orange', alpha=0.2, label=f'Error > +3dB')
            if np.any(clipped_spectral_error < -3):
                ax2.fill_between(freqs, fixed_error_ylim[0], -3,
                                 where=(clipped_spectral_error < -3) & display_freq_mask,
                                 color='lightblue', alpha=0.2, label=f'Error < -3dB')

            ax2.set_title(f'Output {i + 1} - Spectral Error (Avg Abs={mean_abs_spectral_error_eval_range:.1f} dB)',
                          fontsize=plt.rcParams['axes.titlesize'])
            ax2.set_xlabel('Frequency [Hz]', fontsize=plt.rcParams['axes.labelsize'])
            ax2.set_ylabel('Amplitude Error (dB)', fontsize=plt.rcParams['axes.labelsize'])
            ax2.legend(loc='upper right', fontsize=plt.rcParams['legend.fontsize'])
            ax2.grid(True, linestyle=':', alpha=0.6, linewidth=0.05)
            ax2.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
            ax2.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])

            log_str += (f"- Test Output {i + 1} Spectral Analysis:\n"
                        f"  • RMSE: {np.sqrt(np.mean(residual ** 2)):.6f}\n"
                        f"  • Mean Abs Spectral Error (1-{eval_freq_range_test[1]}Hz): {mean_abs_spectral_error_eval_range:.6f} dB\n"
                        f"  • PSD High-freq Attenuation(>100Hz): {hf_attenuation_db:.1f} dB (Pred - True)\n"
                        f"  • Peak Frequency Offset: {f_true[np.argmax(Pxx_true)] - f_pred[np.argmax(Pxx_pred)]:.2f} Hz\n")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        overall_test_avg_abs_spectral_error = np.nanmean(test_overall_spectral_error_sum) if len(
            test_overall_spectral_error_sum) > 0 else np.nan

        fig_test_freq.suptitle(
            f"Testing Spectral Analysis (Best Model) - Overall Avg Abs Spectral Error = {overall_test_avg_abs_spectral_error:.1f} dB",
            fontsize=plt.rcParams['figure.titlesize'])

        combined_spectral_path = os.path.join(save_dir, 'spectral_analysis_all_test_channels_best_model.png')
        plt.savefig(combined_spectral_path, bbox_inches='tight', dpi=300)
        plt.close(fig_test_freq)

        print(f"Aggregated test spectral analysis plot saved to: {combined_spectral_path}")
        print(f"Test Frequency Domain Error (1-{eval_freq_range_test[1]}Hz):")
        for i, mae in enumerate(test_spectral_errors_per_output):
            log_str += f"- Test Output {i + 1} Mean Abs Spectral Error (1-{eval_freq_range_test[1]}Hz): {mae:.6f} dB\n"
            print(f"  - Output {i + 1} Mean Abs Spectral Error: {mae:.6f} dB")
        log_str += f"- Test Set Overall Mean Abs Spectral Error (1-{eval_freq_range_test[1]}Hz): {overall_test_avg_abs_spectral_error:.6f} dB\n"
        print(f"Test Set Overall Mean Abs Spectral Error: {overall_test_avg_abs_spectral_error:.6f} dB")
    except Exception as e:
        log_str += f"[Warning] Test spectral analysis failed: {str(e)}\n"
        print(f"Test spectral analysis error: {e}")

    # ================= Save Log File =================
    log_path = os.path.join(save_dir, 'training_log.txt')
    try:
        timestamp = f"[System Time] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        new_log_entry = "\n" + "=" * 50 + "\n" + timestamp + log_str

        existing_log = ""
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                existing_log = f.read()

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(existing_log + new_log_entry)

        print(f"Complete log saved to: {log_path}")
    except Exception as e:
        print(f"Log saving failed: {e}")

    print("=" * 50)
    print("Visualization workflow completed!")

