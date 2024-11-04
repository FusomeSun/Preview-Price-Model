# Preview Price Model

A deep learning-based price prediction model using various neural network architectures including LSTM, Transformer, and Temporal Fusion Transformer.

## Features

- Multiple neural network architectures support
- Automatic feature selection
- Multi-stock training capability
- Custom loss functions for better price prediction
- Comprehensive evaluation metrics
- Support for both single-stock and multi-stock predictions

## Environment Setup

1. Create a new conda environment:

```bash
conda create -f environment.yaml
```
  Or using mannul env creation :

```bash
conda create -n price_pred python=3.8
conda activate price_pred
```

2. Install required packages:

```bash
pip install -r requirment.txt 
```
  Or using mannul install :

```bash
pip install torch
pip install pandas
pip install numpy
pip install scikit-learn
pip install yfinance
pip install hydra-core
pip install matplotlib
pip install seaborn
pip install selenium
pip install webdriver-manager
```

## Project Structure

```
├── configs/
│   └── default.yaml      # Configuration file
├── data/
│   └── features/         # Feature processing scripts
├── model/
│   ├── LSTM.py          # LSTM model implementations
│   ├── Transformer.py   # Transformer model implementations
│   ├── CNN.py           # CNN model implementations
│   └── utils/           # Model utilities
├── results/             # Saved results and metrics
├──training/
└──  ├──  pre_train.py
     └── fine_tune.py         # Main training script
```

## Usage

### Single Stock Training

1. Modify the configuration in `configs/default.yaml`:
```yaml
stock_info:
  stock_name: NVDA  # Change to your desired stock
  start_date: '2018-01-01'
  end_date: '2024-07-10'
  feature: Close    # Target feature to predict
```

2. Run training:
```bash
python fine_tune.py
```

### Multi-Stock Training

1. Modify the stock list in configuration:
```yaml
stock_info:
  stock_names: 
    - NVDA
    - AAPL
    # Add more stocks as needed
```

2. Run multi-stock training:
```bash
python fine_tune_multi_stocks.py
```

## Configuration Options

### Model Selection
Choose from available models:
- `TransformerModel`
- `TemporalFusionTransformer`

```yaml
fine_tune:
  model: TemporalFusionTransformer  # Change model here
```

### Feature Selection
- `auto_feature: True` - Automatically select best features
- `full_feature: True` - Use all available features
- Or specify custom features in `features` list

### Training Parameters
Adjust in config file:
```yaml
fine_tune:
  epochs: 100
  lr: 0.001
  loss_alpha: 0.2
```

## Results

The model saves:
- Prediction results in CSV format
- Model configuration
- Performance metrics
- Summary statistics

Results are saved in the `results` directory with timestamp-based folders.

## Notes

- The model uses daily data from Yahoo Finance
- Predictions are made for the next N days (configurable in `out_length`)
- Historical data length is configurable via `seq_length`
- Multiple technical indicators are calculated automatically

## Troubleshooting

1. If you encounter CUDA errors, ensure your PyTorch installation matches your CUDA version
2. For memory issues, try reducing batch size or sequence length
3. For Yahoo Finance access issues, ensure stable internet connection

