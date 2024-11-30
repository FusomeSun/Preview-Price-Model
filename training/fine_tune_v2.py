import sys
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import hydra
from omegaconf import DictConfig, OmegaConf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.utils.load import load_model
from data.features.processing import load_stock_data
from results.evaluation import calculate_metrics, print_metrics, calculate_overall_metrics

def load_saved_model(model_dir, cfg):
    """Load the saved model from the specified directory"""
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'))
   
    # Initialize model with current config
    model = load_model(
        cfg.fine_tune.model,
        checkpoint['input_size'],
        checkpoint['output_size'],
        checkpoint['output_length'],
        cfg
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

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
    
def get_date_ranges(end_date, seq_length):
    """Calculate the correct date ranges for data loading and prediction verification"""
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Calculate start date (seq_length trading days before end_date)
    # Multiply by 1.5 to account for weekends and holidays
    start_date = end_date - timedelta(days=int(seq_length * 1.5))
    
    # Calculate prediction end date (out_length days after end_date)
    pred_end_date = end_date + timedelta(days=30)  # Adding extra days to ensure we get enough future data
    
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), pred_end_date.strftime("%Y-%m-%d")

@hydra.main(version_base="1.2", config_path='../configs', config_name='single')
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    # Load the trained model
    model_dir = os.path.join(project_root, 'save_model', cfg.experiment.id)
    model = load_saved_model(model_dir, cfg)  # Pass cfg to load_saved_model
    model = model.to(device)
    model.eval()

    # Calculate date ranges
    start_date, end_date, pred_end_date = get_date_ranges(
        cfg.stock_info.end_date, 
        cfg.hyperparameters.seq_length
    )
    
    print(f"Loading data from {start_date} to {end_date}")
    print(f"Making predictions from {end_date} to {pred_end_date}")

    results = {}
    all_metrics = []

     # Fetch S&P 500 data
    sp500_data = get_sp500_data(cfg.stock_info.end_date, cfg.hyperparameters.out_length)
    
    for stock_name in cfg.stock_info.stock_names:
        try:
            print(f"\nProcessing {stock_name}")
            
            # Load data for the sequence length period
            stock_data = load_stock_data(
                stock_name, 
                start_date,
                end_date
            )
            
            if stock_data is None or stock_data.empty:
                print(f"No data available for {stock_name}")
                continue

            # Prepare input data using the last seq_length days
            if cfg.fine_tune.full_feature:
                predict_data = stock_data[cfg.stock_info.training_features]
            else:
                predict_data = stock_data[cfg.stock_info.features]

            # Make predictions
            with torch.no_grad():
                x_test = torch.tensor(
                    predict_data[-cfg.hyperparameters.seq_length:].values, 
                    dtype=torch.float32
                ).unsqueeze(0).to(device)
                predictions = model(x_test).squeeze().cpu().numpy()

            # Get actual future prices for verification
            future_data = yf.download(
                stock_name, 
                start=end_date,
                end=pred_end_date
            )[cfg.stock_info.feature]
            
            if future_data.empty:
                print(f"No future price data available for {stock_name}")
                continue
                
            actual_prices = future_data.values[:cfg.hyperparameters.out_length]

            # Adjust predictions
            scaler = MinMaxScaler().fit(
                stock_data[cfg.stock_info.feature].values.reshape(-1, 1)
            )
            predictions = scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()[:len(actual_prices)]
            
            difference = actual_prices[0] - predictions[0]
            predictions_adjusted = predictions + difference

            # Calculate metrics
            metrics = calculate_metrics(actual_prices, 
                                     predictions_adjusted, 
                                     sp500_data[:cfg.hyperparameters.out_length])
            all_metrics.append(metrics)

            results[stock_name] = {
                'actual_prices': actual_prices.tolist(),
                'predictions': predictions_adjusted.tolist(),
                'metrics': metrics
            }

            print(f"\nMetrics for {stock_name}:")
            print_metrics(metrics)

        except Exception as e:
            print(f"Error processing predictions for {stock_name}: {e}")
            continue

    if all_metrics:
        overall_metrics = calculate_overall_metrics(all_metrics,cfg.stock_info.stock_names)
        print("\nOverall metrics:")
        print_metrics(overall_metrics)
    else:
        print("\nNo successful predictions to calculate overall metrics")
        overall_metrics = {}
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if(cfg.experiment.Uid):
        results_dir = os.path.join(project_root, 'results', cfg.experiment.id, cfg.experiment.save_id,cfg.fine_tune.prediction_file, timestamp)
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