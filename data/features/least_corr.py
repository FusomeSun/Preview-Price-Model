import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import hydra
from omegaconf import DictConfig

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from features.processing import load_stock_data

def generate_noise_data(data_shape, noise_level=0.05):
    np.random.seed(42)  # For reproducibility
    noise_data = pd.DataFrame(np.random.normal(0, noise_level, size=data_shape), 
                              columns=[f'Noise_{i}' for i in range(data_shape[1])],
                              index=range(data_shape[0]))
    return noise_data

def calculate_spearman_correlation_with_pvalue(data, target):
    correlation_dict = {}
    p_value_dict = {}

    for column in data.columns:
        corr, p_value = stats.spearmanr(data[column], target)
        correlation_dict[column] = corr
        p_value_dict[column] = p_value

    return correlation_dict, p_value_dict

def find_top_least_correlated_combinations(correlation_dict, top_n=5):
    features = list(correlation_dict.keys())
    all_combos = []

    for r in range(5, len(features) + 1):
        combos = combinations(features, r)
        for combo in combos:
            correlation = np.mean([abs(correlation_dict[feature]) for feature in combo])
            all_combos.append((combo, correlation))

    # Sort by absolute correlation and get top_n least correlated
    top_least_correlated = sorted(all_combos, key=lambda x: x[1])[:top_n]
    return top_least_correlated

@hydra.main(version_base="1.1", config_path='../../configs', config_name='tech')
def main(cfg: DictConfig) -> None:
    # Load stock data
    stock_data = load_stock_data(
        cfg.stock_info.stock_name,
        cfg.stock_info.start_date,
        cfg.stock_info.end_date
    )
    
    # Get the last n days of data
    n_days = 3600 # Add this to your config file
    #stock_data = stock_data.iloc[-n_days:]
    
    print(f"Analyzing the last {n_days} days of data")
    
    # Generate noise data for correlation validation
    noise_data = generate_noise_data((len(stock_data), len(stock_data.columns) - 1))  # -1 to exclude 'Close'

    # Calculate correlation between noise features and actual Close price
    noise_correlation_dict, noise_p_value_dict = calculate_spearman_correlation_with_pvalue(noise_data, stock_data['Close'])

    print("\nValidation: Correlation between noise features and Close price:")
    for feature, correlation in noise_correlation_dict.items():
        p_value = noise_p_value_dict[feature]
        print(f"{feature}: correlation = {correlation:.4f}, p-value = {p_value:.15f}")

    # Calculate Spearman correlations and p-values for actual stock data
    actual_correlation_dict, actual_p_value_dict = calculate_spearman_correlation_with_pvalue(
        stock_data.drop('Close', axis=1), stock_data['Close'])

    print("\nCorrelation between actual features and Close price:")
    for feature, correlation in actual_correlation_dict.items():
        p_value = actual_p_value_dict[feature]
        print(f"{feature}: correlation = {correlation:.4f}, p-value = {p_value:.15f}")

    # Find top 5 least correlated feature combinations
    top_least_correlated = find_top_least_correlated_combinations(actual_correlation_dict, top_n=5)
    
    print("\nTop 5 least correlated feature combinations with Close price:")
    for combo, correlation in top_least_correlated:
        features = ', '.join(combo)
        p_values = [f"{actual_p_value_dict[feature]:.15f}" for feature in combo]
        print(f"Features: {features}")
        print(f"Mean absolute correlation: {correlation:.4f}")
        print(f"P-values: {', '.join(p_values)}")
        print()

    # Visualize the correlation matrix
    plt.figure(figsize=(15, 13))
    sns.heatmap(stock_data.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f"Spearman Correlation Matrix for {cfg.stock_info.stock_name} (Last {n_days} days)")
    plt.tight_layout()
    
    # Save the plot
    plot_dir = os.path.join(project_root, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{cfg.stock_info.stock_name}_spearman_correlation_matrix_{n_days}days.png'))
    plt.close()

    print(f"Correlation matrix plot saved to {plot_dir}")

if __name__ == "__main__":
    main()