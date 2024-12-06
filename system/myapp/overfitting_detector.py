import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
import threading
from neptune.types import File
import neptune

class OverfittingDetector:
    def __init__(self, history, model, X_train, y_train, X_val, y_val, neptune_run=None):
        self.history = history
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.neptune_run = neptune_run
        self._lock = threading.Lock()
        
    def _create_figure(self):
        """Thread-safe figure creation"""
        with self._lock:
            return plt.figure(figsize=(12, 6))
            
    def _save_plot(self, fig, name):
        """Thread-safe plot saving"""
        with self._lock:
            if self.neptune_run:
                try:
                    # Upload directly using neptune.types.File.as_image with the figure
                    self.neptune_run[name].upload(neptune.types.File.as_image(fig))
                except Exception as e:
                    print(f"Error saving plot {name}: {e}")
            plt.close(fig)

    def _plot_training_curves(self):
        """Plot training and validation loss curves"""
        fig = self._create_figure()
        with self._lock:
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        self._save_plot(fig, "overfitting/training_curves")

    def _plot_prediction_scatter(self):
        """Plot scatter plot of predictions vs actual values"""
        fig = self._create_figure()
        with self._lock:
            train_pred = self.model.predict(self.X_train, verbose=0)
            val_pred = self.model.predict(self.X_val, verbose=0)
            
            plt.scatter(self.y_train, train_pred, alpha=0.5, label='Training Data')
            plt.scatter(self.y_val, val_pred, alpha=0.5, label='Validation Data')
            plt.plot([min(self.y_train), max(self.y_train)], 
                    [min(self.y_train), max(self.y_train)], 
                    'r--', label='Perfect Prediction')
            plt.title('Predicted vs Actual Values')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.legend()
        self._save_plot(fig, "overfitting/prediction_scatter")

    def _plot_residuals(self):
        """Plot residuals distribution"""
        fig = self._create_figure()
        with self._lock:
            train_pred = self.model.predict(self.X_train, verbose=0)
            val_pred = self.model.predict(self.X_val, verbose=0)
            
            train_residuals = self.y_train - train_pred.flatten()
            val_residuals = self.y_val - val_pred.flatten()
            
            plt.hist(train_residuals, alpha=0.5, label='Training Residuals', bins=30)
            plt.hist(val_residuals, alpha=0.5, label='Validation Residuals', bins=30)
            plt.title('Residuals Distribution')
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.legend()
        self._save_plot(fig, "overfitting/residuals")

    def _plot_loss_distribution(self):
        """Plot loss distribution"""
        fig = self._create_figure()
        with self._lock:
            train_pred = self.model.predict(self.X_train, verbose=0)
            val_pred = self.model.predict(self.X_val, verbose=0)
            
            train_losses = np.square(self.y_train - train_pred.flatten())
            val_losses = np.square(self.y_val - val_pred.flatten())
            
            plt.hist(train_losses, alpha=0.5, label='Training Losses', bins=30)
            plt.hist(val_losses, alpha=0.5, label='Validation Losses', bins=30)
            plt.title('Loss Distribution')
            plt.xlabel('Loss Value')
            plt.ylabel('Frequency')
            plt.legend()
        self._save_plot(fig, "overfitting/loss_distribution")

    def _calculate_metrics(self):
        """Calculate various metrics for overfitting detection"""
        train_pred = self.model.predict(self.X_train, verbose=0)
        val_pred = self.model.predict(self.X_val, verbose=0)
        
        train_mse = mean_squared_error(self.y_train, train_pred)
        val_mse = mean_squared_error(self.y_val, val_pred)
        
        train_mae = mean_absolute_error(self.y_train, train_pred)
        val_mae = mean_absolute_error(self.y_val, val_pred)
        
        return {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'mse_ratio': val_mse / train_mse if train_mse > 0 else float('inf'),
            'mae_ratio': val_mae / train_mae if train_mae > 0 else float('inf')
        }

    def _detect_overfitting(self, metrics, threshold):
        """Detect overfitting based on metrics"""
        mse_ratio = metrics['val_mse'] / metrics['train_mse'] if metrics['train_mse'] > 0 else float('inf')
        mae_ratio = metrics['val_mae'] / metrics['train_mae'] if metrics['train_mae'] > 0 else float('inf')
        return mse_ratio > (1 + threshold) or mae_ratio > (1 + threshold)

    def _log_to_neptune(self, metrics, is_overfit):
        """Log metrics to Neptune"""
        if self.neptune_run:
            for name, value in metrics.items():
                self.neptune_run[f"metrics/{name}"] = value
            self.neptune_run["metrics/is_overfit"] = is_overfit

    def analyze_overfitting(self, threshold=0.1):
        """Comprehensive overfitting analysis"""
        try:
            print("\nCalculating overfitting metrics...")
            metrics = self._calculate_metrics()
            
            print("Generating analysis plots...")
            plot_functions = {
                'training_curves': self._plot_training_curves,
                'prediction_scatter': self._plot_prediction_scatter,
                'residuals': self._plot_residuals,
                'loss_distribution': self._plot_loss_distribution
            }
            
            for name, func in plot_functions.items():
                try:
                    print(f"Generating {name} plot...")
                    func()
                except Exception as e:
                    print(f"Error in {name} plot: {str(e)}")
                    plt.close('all')
                    continue
            
            is_overfit = self._detect_overfitting(metrics, threshold)
            self._log_to_neptune(metrics, is_overfit)
            
            return is_overfit, metrics
            
        except Exception as e:
            print(f"Overfitting analysis failed with error: {str(e)}")
            plt.close('all')
            return False, {}