import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import hydra
from omegaconf import DictConfig, OmegaConf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.utils.loss import CustomLossM
from model.utils.load import load_model
from data.features.processing import create_sequences
from data.utils.sp500 import SP500DataManager

def prepare_data_for_pretraining(all_stock_data, seq_length, out_length):
    X_all, y_all = [], []
    common_features = set(all_stock_data[list(all_stock_data.keys())[0]].columns)

    for ticker, data in all_stock_data.items():
        common_features = common_features.intersection(set(data.columns))

    print(f"Common features across all stocks: {common_features}")

    for ticker, data in all_stock_data.items():
        data = data[list(common_features)]  # Keep only common features
        data = data.ffill().bfill()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        X, y = create_sequences(scaled_data, seq_length, out_length)
        X_all.append(X)
        y_all.append(y)
    
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    
    return X_all, y_all

@hydra.main(version_base="1.1", config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    # Load configuration
    actual_config_name = cfg.get('config_file', 'default') 
    config_path = os.path.join(hydra.utils.get_original_cwd(), f'configs/{actual_config_name}.yaml')
    cfg = OmegaConf.load(config_path)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize data manager and update data
    data_manager = SP500DataManager()
    data_manager.update_data()

    # Load S&P 500 data
    all_stock_data = data_manager.get_data(cfg.stock_info.start_date, cfg.stock_info.end_date)

     # Prepare data for pre-training
    X_all, y_all = prepare_data_for_pretraining(all_stock_data, cfg.hyperparameters.seq_length, cfg.hyperparameters.out_length)


    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_all, dtype=torch.float32).to(device)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=cfg.pretrain.batch_size, shuffle=True)

    # Initialize model
    model = load_model(
        cfg.pretrain.model,
        input_size=X_tensor.shape[2],
        output_size=y_tensor.shape[2],
        output_length=cfg.hyperparameters.out_length,
        cfg=cfg
    )
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pretrain.lr)

    # Pre-training loop
    for epoch in range(cfg.pretrain.epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{cfg.pretrain.epochs}], Loss: {avg_loss:.4f}')

    # Save pre-trained model
    torch.save(model.state_dict(), cfg.pretrain.trained_save_path)
    print(f"Pre-trained model saved to {cfg.pretrain.trained_save_path}")

if __name__ == '__main__':
    main()