import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import random
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.utils.loss import *
from model.utils.load import load_model
from data.features.processing import create_sequences, load_stock_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@hydra.main(version_base="1.1", config_path='../configs', config_name='pre_train')
def main(cfg: DictConfig) -> None:
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    # Create save directories
    model_dir = os.path.join(project_root, 'save_model')
    os.makedirs(model_dir, exist_ok=True)

    model_save_dir = os.path.join(model_dir,cfg.experiment.id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(model_save_dir, exist_ok=True)

    # Collect training data from all stocks
    all_X = []
    all_y = []
    scalers = {}

    for stock_name in cfg.stock_info.stock_names:
        try:
            print(f"\nProcessing {stock_name} for training data")
            stock_data = load_stock_data(stock_name, cfg.stock_info.start_date, cfg.stock_info.end_date)
            
            if stock_data is None or stock_data.empty:
                print(f"No data available for {stock_name}")
                continue
           
            if cfg.fine_tune.full_feature:
                print("training data using all features in config file")
                train_data = stock_data[cfg.stock_info.training_features]
            else:
                print("training data using selected features in config file")
                train_data = stock_data[cfg.stock_info.features]
            feature_names = list(train_data.columns)
            feature_index = feature_names.index(cfg.stock_info.feature)

            scaler = QuantileTransformer(output_distribution='normal')
            scaled_data = scaler.fit_transform(train_data)
            scalers[stock_name] = scaler

            training_data_len = len(scaled_data) - cfg.hyperparameters.seq_length
            data = scaled_data[:training_data_len]

            X_specific, y_specific = create_sequences(data, 
                                                    cfg.hyperparameters.seq_length, 
                                                    cfg.hyperparameters.out_length)
            y_specific = y_specific[:, :, feature_index].reshape(-1, cfg.hyperparameters.out_length, 1)

            all_X.append(X_specific)
            all_y.append(y_specific)

        except Exception as e:
            print(f"Error processing {stock_name} for training: {e}")
            continue

    # Combine and prepare data
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    # Split data
    num_samples = X_combined.shape[0]
    train_size = int(0.8 * num_samples)
    indices = np.random.permutation(num_samples)
    
    X_train = X_combined[indices[:train_size]]
    y_train = y_combined[indices[:train_size]]
    X_val = X_combined[indices[train_size:]]
    y_val = y_combined[indices[train_size:]]

    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), 
                                 torch.FloatTensor(y_train).to(device))
    val_dataset = TensorDataset(torch.FloatTensor(X_val).to(device), 
                               torch.FloatTensor(y_val).to(device))
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.fine_tune.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.fine_tune.batch_size, shuffle=False)

    # Initialize model
    model = load_model(
        cfg.fine_tune.model,
        input_size=X_combined.shape[2],
        output_size=1,
        output_length=cfg.hyperparameters.out_length,
        cfg=cfg
    )
    model = model.to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.fine_tune.lr)
    criterion = globals()[cfg.fine_tune.loss_fun](cfg.fine_tune.loss_alpha)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    # Training loop parameters
     # Training loop parameters
    best_val_loss = float('inf')
    best_model_state = None
    patience = 30
    no_improve_epochs = 0
    consecutive_good_acc = 0
    required_good_epochs = 10
    acc_threshold_reached = False
    accuracy_history = []

    # Training loop
    for epoch in range(cfg.fine_tune.epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # Training phase
        for seq, labels in train_dataloader:
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

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = correct_predictions / total_predictions

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for seq, labels in val_dataloader:
                y_pred = model(seq)
                loss = criterion(y_pred, labels)
                val_loss += loss.item()
                
                pred_direction = (y_pred[:, -1, 0] > y_pred[:, 0, 0]).float()
                true_direction = (labels[:, -1, 0] > labels[:, 0, 0]).float()
                val_correct += (pred_direction == true_direction).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = val_correct / val_total
        accuracy_history.append(val_accuracy)

        # Check accuracy threshold and save best model
        if val_accuracy >= 0.8:  # Threshold for good accuracy
            consecutive_good_acc += 1
            if consecutive_good_acc >= required_good_epochs and not acc_threshold_reached:
                print(f"\nStable validation accuracy threshold of 65% reached at epoch {epoch}")
                acc_threshold_reached = True
                best_val_loss = float('inf')  # Reset best loss to start fresh after reaching threshold
        else:
            consecutive_good_acc = 0

        # Only consider early stopping after reaching accuracy threshold
        if acc_threshold_reached:
            if avg_val_loss < best_val_loss:
                print(f"New best validation loss: {avg_val_loss:.8f}")
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'(Patience counted after maintaining 65% accuracy for {required_good_epochs} epochs)')
                break
        
        # Print status
        print(f'Epoch: {epoch:3} Train Loss: {avg_train_loss:.8f} Train Acc: {train_accuracy:.4f} ' +
              f'Val Loss: {avg_val_loss:.8f} Val Acc: {val_accuracy:.4f} ' +
              f'Consecutive Good Epochs: {consecutive_good_acc}/{required_good_epochs}')

        scheduler.step(avg_val_loss)

    # Only save model if accuracy threshold was reached
    if acc_threshold_reached and best_model_state is not None:
        # Save the best model
        model_save_path = os.path.join(model_save_dir, 'best_model.pth')
        torch.save({
            'model_state_dict': best_model_state,
            'model_name': cfg.fine_tune.model,
            'input_size': X_combined.shape[2],
            'output_size': 1,
            'output_length': cfg.hyperparameters.out_length,
            'final_accuracy': val_accuracy,
            'best_val_loss': best_val_loss
        }, model_save_path)

        # Save scalers
        scaler_save_path = os.path.join(model_save_dir, 'scalers.pkl')
        torch.save(scalers, scaler_save_path)

        # Save configuration
        config_save_path = os.path.join(model_save_dir, 'config.yaml')
        OmegaConf.save(cfg, config_save_path)

        # Save training metadata with accuracy information
        metadata = {
            'timestamp': timestamp,
            'final_val_loss': float(best_val_loss),
            'final_val_accuracy': float(val_accuracy),
            'accuracy_threshold_reached': True,
            'consecutive_good_epochs_required': required_good_epochs,
        }
        metadata_path = os.path.join(model_save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"\nModel and related files saved in: {model_save_dir}")
    else:
        print("\nModel did not reach the required accuracy threshold or failed to improve. No model saved.")

if __name__ == '__main__':
    main()