import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score, recall_score, accuracy_score

def calculate_daily_returns(prices):
    """Calculate daily returns from price series."""
    return np.diff(prices) / prices[:-1]

def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Calculate the Sharpe Ratio for a series of returns.
    
    Args:
        returns (np.array): Array of returns
        risk_free_rate (float): Annual risk-free rate (default 2%)
        periods_per_year (int): Number of periods in a year (252 for daily data)
    
    Returns:
        float: Sharpe Ratio
    """
    if len(returns) < 2:
        return 0
    
    # Annualize return and risk-free rate
    avg_return = np.mean(returns) * periods_per_year
    daily_rf_rate = (1 + risk_free_rate) ** (1/periods_per_year) - 1
    excess_return = avg_return - daily_rf_rate
    
    # Annualize volatility
    volatility = np.std(returns) * np.sqrt(periods_per_year)
    
    # Calculate Sharpe Ratio
    sharpe_ratio = excess_return / volatility if volatility != 0 else 0
    
    return sharpe_ratio

def calculate_metrics(actual, predicted_adjusted, sp500_actual, tolerance=3):
    """
    Calculate various evaluation metrics for the predictions.
    
    Args:
        actual (np.array): Array of actual values
        predicted (np.array): Array of predicted values
        sp500_actual (np.array): Array of S&P 500 values
        tolerance (float): Tolerance for price accuracy (default 3%)
    
    Returns:
        dict: Dictionary containing the calculated metrics
    """
    # Direction accuracy
    actual_direction = (actual[1:] > actual[:-1]).astype(int)
    predicted_direction = (predicted_adjusted[1:] > predicted_adjusted[:-1]).astype(int)
    correct_direction_mask = (actual_direction == predicted_direction)
    actual_trend = actual[-1] - actual[0]
    predicted_trend = predicted_adjusted[-1] - predicted_adjusted[0]
    
    # Calculate returns
    actual_returns = calculate_daily_returns(actual)
    strategy_returns = np.where(correct_direction_mask, actual_returns, 0)
   
    sp500_returns = calculate_daily_returns(sp500_actual) if sp500_actual is not None else None
    
    # Calculate Sharpe Ratios
    actual_sharpe = calculate_sharpe_ratio(actual_returns)
    predicted_sharpe = calculate_sharpe_ratio(strategy_returns)
    sp500_sharpe = calculate_sharpe_ratio(sp500_returns) if sp500_returns is not None else None
    
    # Original metrics
    return_rate = (actual[-1] - actual[0]) / actual[0]
    return_rate_abs = abs((actual[-1] - actual[0]) / actual[0])
    direction_accuracy = accuracy_score(actual_direction, predicted_direction)
    n_to_0_directional_accuracy = 1 if np.sign(actual_trend) == np.sign(predicted_trend) else 0
    mse = mean_squared_error(actual, predicted_adjusted)
    mse_rate = np.sqrt(mse) / np.mean(actual) * 100
    precision = precision_score(actual_direction, predicted_direction)
    recall = recall_score(actual_direction, predicted_direction)
    
    # Trend calculations
    actual_trend = actual[-1] - actual[0]
    predicted_trend = predicted_adjusted[-1] - predicted_adjusted[0]
    positive_predict_rate = (actual_trend/actual[0]) if (predicted_trend > 0) else 0
    negative_predict_rate = -1*(actual_trend/actual[0]) if (predicted_trend < 0) else 0
    total_return_rate = positive_predict_rate + negative_predict_rate
    
    # S&P 500 calculations
    sp500_return_rate = (sp500_actual[-1] - sp500_actual[0]) / sp500_actual[0] if sp500_actual is not None else None
    
   
    
    # Add S&P 500 metrics if available
    if sp500_actual is not None:
         metrics = {
            'Raw Positive Price change (%)': return_rate * 100,
            'Raw total Price change (%)': return_rate_abs * 100,
            'Total Return Rate': total_return_rate * 100,
            'Positive Predict Rate (%)': positive_predict_rate * 100,
            'Negative Predict Rate (%)': negative_predict_rate * 100,
            'S&P 500 index Return Rate (%)': sp500_return_rate * 100,
            'Direction Accuracy (%)': direction_accuracy * 100,
            'N_to_0 Direction Accuracy (%)' : n_to_0_directional_accuracy *100,
            'MSE Rate (%)': mse_rate,
            'Precision (%)': precision * 100,
            'Recall (%)': recall * 100,
            'Actual Sharpe Ratio': actual_sharpe,
            'Predicted Sharpe Ratio': predicted_sharpe,
            'S&P 500 Sharpe Ratio': sp500_sharpe
        }
    else :

         metrics = {
            'Raw Positive Price change (%)': return_rate * 100,
            'Raw total Price change (%)': return_rate_abs * 100,
            'Total Return Rate': total_return_rate * 100,
            'Positive Predict Rate (%)': positive_predict_rate * 100,
            'Negative Predict Rate (%)': negative_predict_rate * 100,
            'Direction Accuracy (%)': direction_accuracy * 100,
            'N_to_0 Direction Accuracy (%)' : n_to_0_directional_accuracy *100,
            'MSE Rate (%)': mse_rate,
            'Precision (%)': precision * 100,
            'Recall (%)': recall * 100,
            'Actual Sharpe Ratio': actual_sharpe,
            'Predicted Sharpe Ratio': predicted_sharpe
        }

    return metrics

def print_metrics(metrics):
    """
    Print the evaluation metrics in a formatted manner.
    
    Args:
        metrics (dict): Dictionary containing the calculated metrics
    """
    for metric, value in metrics.items():
        if 'Sharpe Ratio' in metric:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value:.2f}")

def calculate_overall_metrics(all_metrics):
    """
    Calculate overall metrics across all stocks.
    
    Args:
        all_metrics (list): List of dictionaries containing metrics for each stock
        
    Returns:
        dict: Dictionary containing the averaged metrics
    """
    overall_metrics = {}
    sum_metrics = [
        'Raw Positive Price change (%)',
        'Raw total Price change (%)',
        'Total Return Rate',
        'Positive Predict Rate (%)',
        'Negative Predict Rate (%)'
    ]
    
    for metric in all_metrics[0].keys():
        # Get all valid values for this metric
        valid_values = [stock_metrics[metric] for stock_metrics in all_metrics 
                       if metric in stock_metrics and stock_metrics[metric] is not None]
        
        if valid_values:  # Only calculate if we have valid values
            if metric in sum_metrics:
                overall_metrics[metric] = np.sum(valid_values) / len(all_metrics)
            else:
                overall_metrics[metric] = np.mean(valid_values)
                
    return overall_metrics