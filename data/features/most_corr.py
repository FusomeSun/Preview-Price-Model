import sys
import os
import pandas as pd
import numpy as np
from itertools import combinations
import hydra
from omegaconf import DictConfig
import seaborn as sns
import matplotlib.pyplot as plt


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from features.processing import load_stock_data

def prepare_data_for_correlation(data, target_column='Close', forecast_horizon=10):
    for i in range(1, forecast_horizon + 1):
        data[f'Future_Close_{i}'] = data[target_column].shift(-i)
    return data.dropna()

def calculate_correlations(data, target_columns):
    correlation_matrix = data.corr()
    future_correlations = correlation_matrix[target_columns].mean(axis=1).sort_values(ascending=False)
    return correlation_matrix

def explore_feature_combinations(data, target_columns, max_features=8):
    def get_combined_correlation(df, features):
        valid_features = [f for f in features if f in df.columns]
        combined_feature = df[valid_features].mean(axis=1)
        return df[target_columns].corrwith(combined_feature).mean()

    all_features = [col for col in data.columns if col not in target_columns and col != 'Close']
    
    print("Available features for combinations:")
    print(all_features)
    
    all_correlations = []
    
    for n in range(1, max_features + 1):
        feature_combos = list(combinations(all_features, n))
        correlations = [(combo, get_combined_correlation(data, combo), n) for combo in feature_combos]
        all_correlations.extend(correlations)
    
    # Sort all combinations by absolute correlation value
    sorted_combinations = sorted(all_correlations, key=lambda x: x[1], reverse=True)
    
    # Get top 20 combinations overall
    top_5 = sorted_combinations[:5]
    
    print("\nTop 5 combinations overall:")
    for combo, corr, n in top_5:
        print(f"{combo} ({n} features): {corr}")
    combo,corr,_ = top_5[0]
    return combo,corr
@hydra.main(version_base="1.1", config_path='../../configs', config_name='tech')
def main(cfg: DictConfig) -> None:
    # Load stock data
    stock_data = load_stock_data(
        cfg.stock_info.stock_name,
        cfg.stock_info.start_date,
        cfg.stock_info.end_date
    )
    
    # Prepare data for correlation analysis
    forecast_horizon = cfg.hyperparameters.out_length
    prepared_data = prepare_data_for_correlation(stock_data,target_column=cfg.stock_info.feature, forecast_horizon=forecast_horizon)

    # Define target columns (future prices)
    target_columns = [f'Future_Close_{i}' for i in range(1, forecast_horizon + 1)]

    # Calculate correlations
    correlation_matrix = calculate_correlations(prepared_data, target_columns)

    # Explore feature combinations
    combo,corr = explore_feature_combinations(prepared_data, target_columns,len(stock_data.columns))
    
    # Visualize the correlation matrix
    plt.figure(figsize=(15, 13))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f"Correlation Matrix for {cfg.stock_info.stock_name} (including future prices)")
    plt.tight_layout()
    
    # Save the plot
    plot_dir = os.path.join(project_root, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{cfg.stock_info.stock_name}_correlation_matrix.png'))
    plt.close()

    print(f"Correlation matrix plot saved to {plot_dir}")

if __name__ == "__main__":
    main()