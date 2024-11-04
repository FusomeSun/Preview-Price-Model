import sys
import os
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
from datetime import datetime
import random
import optuna

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.utils.loss import *
from model.utils.load import load_model
from data.features.processing import create_sequences, load_stock_data
from data.features.most_corr import prepare_data_for_correlation, calculate_correlations, explore_feature_combinations
from data.features.least_corr import calculate_spearman_correlation_with_pvalue, find_top_least_correlated_combinations
from results.evaluation import calculate_metrics, print_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_stock_data(cfg, stock_name, stock_data):
    if cfg.fine_tune.auto_feature:
        actual_correlation_dict, _ = calculate_spearman_correlation_with_pvalue(
            stock_data.drop('Close', axis=1), stock_data['Close'])
        top_least_correlated = find_top_least_correlated_combinations(actual_correlation_dict, top_n=1)
        features, _ = top_least_correlated[0]
        features = list(features) + ['Close']
        train_data = stock_data[features]
    elif cfg.fine_tune.full_feature:
        train_data = stock_data[cfg.stock_info.training_features]
    else:
        train_data = stock_data[cfg.stock_info.features]
    
    feature_names = list(train_data.columns)
    feature_index = feature_names.index(cfg.stock_info.feature)
    
    scaler = QuantileTransformer(output_distribution='normal')
    scaled_data = scaler.fit_transform(train_data)
    
    return scaled_data, feature_index

def objective(trial, cfg, stock_data_dict, device):
    # Suggest hyperparameters
    cfg.hyperparameters.seq_length = trial.suggest_int('seq_length', 100, 180)
    cfg.fine_tune.lr = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    cfg.fine_tune.loss_alpha = trial.suggest_uniform('loss_alpha', 0.5, 0.9)
    
    total_val_loss = 0
    
    for stock_name, stock_data in stock_data_dict.items():
        scaled_data, feature_index = prepare_stock_data(cfg, stock_name, stock_data)
        
        X, y = create_sequences(scaled_data, cfg.hyperparameters.seq_length, cfg.hyperparameters.out_length)
        y = y[:, :, feature_index].reshape(-1, cfg.hyperparameters.out_length, 1)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Create DataLoader
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Load model
        model = load_model(
            cfg.fine_tune.model,
            input_size=X.shape[2],
            output_size=1,
            output_length=cfg.hyperparameters.out_length,
            cfg=cfg
        ).to(device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.fine_tune.lr)
        criterion = globals()[cfg.fine_tune.loss_fun](cfg.fine_tune.loss_alpha)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(cfg.fine_tune.epochs):
            model.train()
            for seq, labels in train_loader:
                seq, labels = seq.to(device), labels.to(device)
                optimizer.zero_grad()
                y_pred = model(seq)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for seq, labels in val_loader:
                    seq, labels = seq.to(device), labels.to(device)
                    y_pred = model(seq)
                    val_loss += criterion(y_pred, labels).item()
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        total_val_loss += best_val_loss
    
    # Return average validation loss across all stocks
    return total_val_loss / len(stock_data_dict)

@hydra.main(version_base="1.1", config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    set_seed(42)

    actual_config_name = cfg.get('config_file', 'default') 
    config_path = os.path.join(hydra.utils.get_original_cwd(), f'configs/{actual_config_name}.yaml')
    cfg = OmegaConf.load(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    # Load stock data for all stocks
    stock_data_dict = {}
    for stock_name in cfg.stock_info.stock_names:
        stock_data_dict[stock_name] = load_stock_data(stock_name, cfg.stock_info.start_date, cfg.stock_info.end_date)

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, cfg, stock_data_dict, device), n_trials=cfg.optimze.num_trail)

    # Print the best parameters and save them to the config file
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        if key in cfg.hyperparameters:
            cfg.hyperparameters[key] = value
        elif key in cfg.fine_tune:
            cfg.fine_tune[key] = value

    # Save the updated config
    OmegaConf.save(cfg, config_path)
    print(f"Updated configuration saved to {config_path}")

if __name__ == '__main__':
    main()