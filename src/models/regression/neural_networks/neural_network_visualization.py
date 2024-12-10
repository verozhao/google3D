import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
import tensorflow as tf

def plot_model_architecture(model):
    """
    Plot and display the model architecture using TensorFlow's model visualization.
    
    Args:
        model: TensorFlow/Keras model instance
    """
    model.summary()
    SVG(tf.keras.utils.model_to_dot(model, dpi=70).create(prog='dot', format='svg'))

def print_training_metrics(history, mse, r2, best_hps):
    """
    Print training and validation metrics.
    
    Args:
        history: Training history object
        mse: Mean squared error on test set
        r2: R-squared score on test set
        best_hps: Best hyperparameters from tuning
    """
    print("\nFinal Training Metrics:")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Number of epochs completed: {len(history.history['loss'])}")

    print("\nTest Set Performance:")
    print(f"Mean Squared Error (log scale): {mse:.4f}")
    print(f"R² Score (log scale): {r2:.4f}")

    print("\nBest Hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

def calculate_error_metrics(y_test, y_pred):
    """
    Calculate various error metrics in both log and original scale.
    
    Args:
        y_test: True values
        y_pred: Predicted values
    """
    mse_log = mean_squared_error(y_test, y_pred)
    print(f"\nMSE (log scale): {mse_log:.4f}")

    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
    mse_original = mean_squared_error(y_test_original, y_pred_original)
    print(f"MSE (original scale): {mse_original:.2f}")

    rmse_log = np.sqrt(mse_log)
    rmse_original = np.sqrt(mse_original)
    print(f"RMSE (log scale): {rmse_log:.4f}")
    print(f"RMSE (original scale): {rmse_original:.2f}")

    print("Shapes:")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_pred shape: {y_pred.shape}")

def calculate_additional_metrics(y_test, y_pred):
    """
    Calculate additional performance metrics including MAE and MAPE.
    
    Args:
        y_test: True values
        y_pred: Predicted values
    """
    y_test_array = np.array(y_test).flatten()
    y_pred_array = np.array(y_pred).flatten()

    mae_log = np.mean(np.abs(y_test_array - y_pred_array))
    y_test_original = np.expm1(y_test_array)
    y_pred_original = np.expm1(y_pred_array)
    mae_original = np.mean(np.abs(y_test_original - y_pred_original))

    print(f"\nMAE (log scale): {mae_log:.4f}")
    print(f"MAE (original scale): {mae_original:.2f}")

    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    print(f"\nMAPE: {mape:.2f}%")

def plot_residual_analysis(y_test, y_pred):
    """
    Create residual analysis plots including residual plot, Q-Q plot, and error distribution.
    
    Args:
        y_test: True values
        y_pred: Predicted values
    """
    y_test_array = np.array(y_test).flatten()
    y_pred_array = np.array(y_pred).flatten()
    residuals = y_test_array - y_pred_array

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(y_pred_array, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values (log scale)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    plt.subplot(1, 3, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')

    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=50, density=True, alpha=0.7)
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2,
             label=f'Normal Dist.\nμ={mu:.2f}, σ={sigma:.2f}')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nResidual Statistics:")
    print(f"Mean of residuals: {np.mean(residuals):.4f}")
    print(f"Standard deviation of residuals: {np.std(residuals):.4f}")
    print(f"Skewness: {stats.skew(residuals):.4f}")
    print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")

def analyze_prediction_errors(y_test, y_pred):
    """
    Analyze prediction errors in terms of actual view counts.
    
    Args:
        y_test: True values
        y_pred: Predicted values
    """
    y_test_array = np.array(y_test).flatten()
    y_pred_array = np.array(y_pred).flatten()
    
    y_test_original = np.expm1(y_test_array)
    y_pred_original = np.expm1(y_pred_array)

    mae_views = np.mean(np.abs(y_test_original - y_pred_original))
    print(f"MAE: {mae_views:,.0f} views")

    errors = np.abs(y_test_original - y_pred_original)
    print("\nError Distribution (views):")
    print(f"25th percentile: {np.percentile(errors, 25):,.0f}")
    print(f"Median error: {np.percentile(errors, 50):,.0f}")
    print(f"75th percentile: {np.percentile(errors, 75):,.0f}")
    print(f"90th percentile: {np.percentile(errors, 90):,.0f}")

def plot_view_distributions(y_test):
    """
    Plot various distributions of view counts.
    
    Args:
        y_test: True values
    """
    y_test_array = np.array(y_test).flatten()
    y_test_original = np.expm1(y_test_array)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(y_test_original, bins=50, alpha=0.7)
    plt.title('View Count Distribution')
    plt.xlabel('Views (millions)')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(y_test_original, bins=50, alpha=0.7, log=True)
    plt.title('View Count Distribution (Log Y-axis)')
    plt.xlabel('Views (millions)')
    plt.ylabel('Frequency (log scale)')

    plt.subplot(1, 3, 3)
    plt.hist(np.log1p(y_test_original), bins=50, alpha=0.7)
    plt.title('Log-Transformed View Distribution')
    plt.xlabel('Log(Views in millions + 1)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    print("\nView Count Statistics:")
    print(f"Mean views: {np.mean(y_test_original):,.0f}")
    print(f"Median views: {np.median(y_test_original):,.0f}")
    print(f"Std deviation: {np.std(y_test_original):,.0f}")
    print("\nPercentiles:")
    percentiles = [5, 25, 50, 75, 95, 99]
    for p in percentiles:
        print(f"{p}th percentile: {np.percentile(y_test_original, p):,.0f} views")

def run_full_visualization(model, history, y_test, y_pred, mse, r2, best_hps):
    """
    Run all visualization and analysis functions in sequence.
    
    Args:
        model: TensorFlow/Keras model instance
        history: Training history object
        y_test: True values
        y_pred: Predicted values
        mse: Mean squared error on test set
        r2: R-squared score on test set
        best_hps: Best hyperparameters from tuning
    """
    plot_model_architecture(model)
    print_training_metrics(history, mse, r2, best_hps)
    calculate_error_metrics(y_test, y_pred)
    calculate_additional_metrics(y_test, y_pred)
    plot_residual_analysis(y_test, y_pred)
    analyze_prediction_errors(y_test, y_pred)
    plot_view_distributions(y_test)
