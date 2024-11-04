import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from datetime import datetime, timedelta
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.utils.loss import *
from model.utils.load import load_model
from data.features.processing import create_sequences, load_data, load_stock_data
from data.features.most_corr import prepare_data_for_correlation, calculate_correlations, explore_feature_combinations
from data.features.least_corr import calculate_spearman_correlation_with_pvalue, find_top_least_correlated_combinations
from results.evaluation import calculate_metrics, print_metrics,calculate_overall_metrics

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_sp500_data(start_date, out_length):
    """
    Fetch S&P 500 data starting from the given date for out_length + 10 days.
    """
    try:
        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=out_length + 10)).strftime("%Y-%m-%d")
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)
        return sp500['Close']
    except Exception as e:
        print(f"Failed to download S&P500 data: {e}")
        return None


    
def process_stock(cfg, stock_name, device, sp500_data):
    FINE_TUNE_EPOCHS = cfg.fine_tune.epochs
    OUT_LENGTH = cfg.hyperparameters.out_length
    SEQ_LENGTH = cfg.hyperparameters.seq_length
  
    try:
        # Load stock data with additional features
        stock_data = load_stock_data(stock_name, cfg.stock_info.start_date, cfg.stock_info.end_date)
        if stock_data is None or stock_data.empty:
            print(f"No data available for {stock_name}")
            return None, None, None, None
            
        
    except Exception as e:
        print(f"Error processing {stock_name}: {e}")
        return None, None, None, None

    if cfg.fine_tune.auto_feature:
        actual_correlation_dict, actual_p_value_dict = calculate_spearman_correlation_with_pvalue(
            stock_data.drop('Close', axis=1), stock_data['Close'])
        top_least_correlated = find_top_least_correlated_combinations(actual_correlation_dict, top_n=1)
        features, corr = top_least_correlated[0]
        features = list(features) + ['Close']
        print(f"{stock_name} features: {features}, correlation: {corr}")
        train_data = stock_data[features]
    elif cfg.fine_tune.full_feature:
        print("training data using all features in congfig file")
        train_data = stock_data[cfg.stock_info.training_features]
    else:
        print("training data using selected  features in congfig file")
        train_data = stock_data[cfg.stock_info.features]

    feature_names = list(train_data.columns)
    feature_index = feature_names.index(cfg.stock_info.feature)
    
    # Prepare data
    scaler = QuantileTransformer(output_distribution='normal')
    scaled_data = scaler.fit_transform(train_data)  

    training_data_len = len(scaled_data) - SEQ_LENGTH
    train_data = scaled_data[:training_data_len]

    # Create sequences
    X_specific, y_specific = create_sequences(train_data, SEQ_LENGTH, OUT_LENGTH)
    y_specific = y_specific[:, :, feature_index].reshape(-1, OUT_LENGTH, 1)

    # Convert to PyTorch tensors
    X_specific_tensor = torch.tensor(X_specific, dtype=torch.float32).to(device)
    y_specific_tensor = torch.tensor(y_specific, dtype=torch.float32).to(device)

    # Create DataLoader
    specific_dataset = TensorDataset(X_specific_tensor, y_specific_tensor)
    specific_dataloader = DataLoader(specific_dataset, batch_size=cfg.fine_tune.batch_size, shuffle=cfg.fine_tune.shuffle)
  
    # Load the model
    model = load_model(
        cfg.fine_tune.model,
        input_size=X_specific_tensor.shape[2],
        output_size=1,
        output_length=cfg.hyperparameters.out_length,
        cfg=cfg
    )
    model = model.to(device)

    # Optimizer for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.fine_tune.lr)
    criterion = globals()[cfg.fine_tune.loss_fun](cfg.fine_tune.loss_alpha)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    best_loss = float('inf')
    patience = 150
    no_improve_epochs = 0
    consecutive_good_acc = 0  # Counter for consecutive epochs with good accuracy
    required_good_epochs = 10  # Number of consecutive epochs needed above 70%
    acc_threshold_reached = False  # Flag for stable accuracy achievement
    best_accuracy = 0.0
    accuracy_history = []  # Track accuracy history for analysis

    # Fine-tuning loop
    for epoch in range(FINE_TUNE_EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        total_mse = 0

        for seq, labels in specific_dataloader:
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred_direction = (y_pred[:, -1, 0] > y_pred[:, 0, 0]).float()
            true_direction = (labels[:, -1, 0] > labels[:, 0, 0]).float()
            correct_predictions += (pred_direction == true_direction).sum().item()
            total_predictions += labels.size(0)

            mse = nn.MSELoss()(y_pred, labels)
            total_mse += mse.item()

        avg_loss = total_loss / len(specific_dataloader)
        accuracy = correct_predictions / total_predictions
        avg_mse = total_mse / len(specific_dataloader)
        accuracy_history.append(accuracy)

        # Update best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Check consecutive accuracy above threshold
        if accuracy >= 0.70:
            consecutive_good_acc += 1
            if consecutive_good_acc >= required_good_epochs and not acc_threshold_reached:
                print(f"\nStable accuracy threshold of 70% reached at epoch {epoch}")
                print(f"Last {required_good_epochs} accuracies: {accuracy_history[-required_good_epochs:]}")
                acc_threshold_reached = True
                no_improve_epochs = 0  # Reset counter when stability is first reached
                best_loss = avg_loss
        else:
            consecutive_good_acc = 0  # Reset counter if accuracy drops below threshold

        # Status update
        print(f'{stock_name} - Epoch: {epoch:3} Loss: {avg_loss:.8f} Accuracy: {accuracy:.4f} MSE: {avg_mse:.8f} ' +
              f'Consecutive Good Epochs: {consecutive_good_acc}/{required_good_epochs}')
        
        scheduler.step(avg_loss)
        
        # Only start counting patience after stable accuracy is reached
        if acc_threshold_reached:
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'(Patience counted after maintaining 70% accuracy for {required_good_epochs} epochs)')
                break
        
    print(f'\nFine-tuning completed for {stock_name}:')
    print(f'Final loss: {avg_loss:.8f}')
    print(f'Final accuracy: {accuracy:.4f}')
    print(f'Best accuracy: {best_accuracy:.4f}')
    print(f'Stable accuracy threshold reached: {acc_threshold_reached}')
    if acc_threshold_reached:
        print(f'Epochs after stability: {epoch - (epoch - consecutive_good_acc)}')
        
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(scaled_data[-cfg.hyperparameters.seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
        predictions = model(x_test).squeeze().cpu().numpy()

    # Inverse transform predictions
    close_price_scaler = MinMaxScaler().fit(stock_data[cfg.stock_info.feature].values.reshape(-1, 1))
    predictions = close_price_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    try:
        data = yf.download(stock_name, start=cfg.stock_info.end_date, end="2025-08-07")
        if data.empty:
            print(f"No actual price data available for {stock_name}.")
            return None, None, None, None

        if cfg.stock_info.feature == 'Amplitude':
            data['Stock Amplitude'] = (data['High'] - data['Low']) / data['Open']
            actual_prices = data['Stock Amplitude'].values
        else:
            actual_prices = data[cfg.stock_info.feature].values
    except Exception as e:
        print(f"Error downloading actual prices for {stock_name}: {e}")
        return None, None, None, None
    actual_prices = actual_prices[:cfg.hyperparameters.out_length]
    
    difference = actual_prices[0] - predictions[0]
    predicted_adjusted = predictions + difference

    print(f"{stock_name} - Actual Prices:", actual_prices)
    print(f"{stock_name} - Adjusted predictions:", predicted_adjusted)
    
    sp500_prices = sp500_data[:cfg.hyperparameters.out_length]
     # Calculate metrics
    metrics = calculate_metrics(actual_prices, predicted_adjusted, sp500_prices)

    return actual_prices, predicted_adjusted, metrics, sp500_prices


@hydra.main(version_base="1.1", config_path='../configs', config_name='single')
def main(cfg: DictConfig) -> None:
    set_seed(42)
    """
    # Get the actual config file name from the meta-config
    actual_config_name = cfg.get('config_file', 'default') 
    config_path = os.path.join(hydra.utils.get_original_cwd(), f'configs/{actual_config_name}.yaml')
    cfg = OmegaConf.load(config_path)
    """
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    # Fetch S&P 500 data
    sp500_data = get_sp500_data(cfg.stock_info.end_date, cfg.hyperparameters.out_length)

    # Process each stock
    results = {}
    all_metrics = []
    for stock_name in cfg.stock_info.stock_names:
        print(f"\nProcessing {stock_name}")
        actual_prices, predictions, metrics, sp500_prices = process_stock(cfg, stock_name, device, sp500_data)
        


        if actual_prices is not None and metrics is not None:
                print(f"\nMetrics for {stock_name}:")
                print_metrics(metrics)
                
                results[stock_name] = {
                    'actual_prices': actual_prices.tolist(),
                    'predictions': predictions.tolist(),
                    'sp500_prices': sp500_prices.tolist() if sp500_prices is not None else None,
                    'metrics': metrics
                }
                all_metrics.append(metrics)
      

    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(all_metrics)
    print("\nOverall metrics:")
    print_metrics(overall_metrics)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if(cfg.experiment.Uid):
        results_dir = os.path.join(project_root, 'results', cfg.experiment.id,cfg.fine_tune.prediction_file, timestamp)
    else:
        results_dir = os.path.join(project_root, 'results', cfg.stock_info.end_date,cfg.fine_tune.prediction_file, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Save predictions and metrics for each stock
    for stock_name, stock_results in results.items():
        stock_df = pd.DataFrame({
            'Date': pd.date_range(start=cfg.stock_info.end_date, periods=len(stock_results['predictions'])),
            'Actual': stock_results['actual_prices'],
            'Predicted': stock_results['predictions']
        })
        predictions_file = os.path.join(results_dir, f"{stock_name}_{cfg.fine_tune.prediction_file}")
        stock_df.to_csv(predictions_file, index=False)
        print(f"Predictions for {stock_name} saved to {predictions_file}")

    # Save configuration
    config_file = os.path.join(results_dir, 'config.txt')
    with open(config_file, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"Configuration saved to {config_file}")
    print(results)
    # Save summary with metrics for all stocks and overall metrics
    summary = {
        'stocks': cfg.stock_info.stock_names,
        'model': cfg.fine_tune.model,
         'individual_metrics': {stock: results[stock]['metrics'] 
                                 for stock in results},
        'overall_metrics': overall_metrics,
        'timestamp': timestamp
    }
    summary_file = os.path.join(results_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}:\n")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            elif isinstance(value, list):
                for item in value:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"  {value}\n")
            f.write("\n")
    print(f"Summary saved to {summary_file}")

if __name__ == '__main__':
    main()