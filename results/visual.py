import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys

def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return yaml.safe_load(content)

def extract_config_info(config):
    return {
        'end_date': config['stock_info']['end_date'],
        'prediction_length': f"{config['hyperparameters']['seq_length']},{config['hyperparameters']['out_length']}",
        'feature_selection': 'auto' if config['fine_tune']['auto_feature'] else 'full' if config['fine_tune']['full_feature'] else 'custom'
    }

def extract_summary_metrics(summary):
    return summary['overall_metrics']

def generate_table(experiment_dir):
    rows = []
    for model_dir in os.listdir(experiment_dir):
        model_path = os.path.join(experiment_dir, model_dir)
        if os.path.isdir(model_path):
            for timestamp_folder in os.listdir(model_path):
                folder_path = os.path.join(model_path, timestamp_folder)
                if os.path.isdir(folder_path):
                    config_path = os.path.join(folder_path, 'config.txt')
                    summary_path = os.path.join(folder_path, 'summary.txt')
                    if os.path.exists(config_path) and os.path.exists(summary_path):
                        try:
                            config = parse_file(config_path)
                            summary = parse_file(summary_path)
                            row = {'model': config['fine_tune']['model']}
                            row.update(extract_config_info(config))
                            row.update(extract_summary_metrics(summary))
                            rows.append(row)
                         
                        except Exception as e:
                            print(f"Error processing {folder_path}: {str(e)}")
    return pd.DataFrame(rows)

def save_table(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'results_table.csv')
    df.to_csv(output_path, index=False)
    print(f"Table saved to {output_path}")

def visualize_results(df):
    models = df['model'].unique()
    feature_methods = df['feature_selection'].unique()
    metrics = ['Direction Accuracy (%)', 'MSE Rate (%)', 'Total Return Rate', 'Excess Return Over S&P 500 (%)']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        for model in models:
            model_data = df[df['model'] == model]
            x = range(len(feature_methods))
            y = [model_data[model_data['feature_selection'] == method][metric].values[0] if metric in model_data.columns else 0 for method in feature_methods]
            ax.plot(x, y, marker='o', label=model)

        # Add S&P 500 baseline for Total Return Rate
        if metric == 'Total Return Rate':
            sp500_column = 'S&P 500 index Return Rate (%)'
            if sp500_column in df.columns:
                sp500_return = df[sp500_column].mean()
                ax.axhline(y=sp500_return, color='r', linestyle='--', label='S&P 500')
            else:
                print(f"Warning: '{sp500_column}' not found in the data. S&P 500 baseline not added.")

        ax.set_title(metric)
        ax.set_xticks(range(len(feature_methods)))
        ax.set_xticklabels(feature_methods)
        ax.set_xlabel('Feature Selection Method')
        ax.set_ylabel('Value')
        ax.legend()

    plt.tight_layout()
    return fig

def save_visualization(fig, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'results_visualization.png')
    fig.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visual.py <experiment_date>")
        sys.exit(1)

    experiment_date = sys.argv[1]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    experiment_dir = os.path.join(project_root, 'results', f'{experiment_date}')

    if not os.path.exists(experiment_dir):
        print(f"Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    df = generate_table(experiment_dir)
    save_table(df, experiment_dir)
    fig = visualize_results(df)
    save_visualization(fig, experiment_dir)