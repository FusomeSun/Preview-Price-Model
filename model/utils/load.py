from model.LSTM import ImprovedLSTM, AttentionLSTM, CNNLSTM,BiLSTM
from model.Transformer import TransformerModel, TemporalFusionTransformer



def load_model(model_name, input_size, output_size, output_length, cfg):
    if model_name == "TransformerModel":
        return TransformerModel(
            input_size=input_size,
            output_size=output_size,
            output_length=output_length,
            d_model=cfg.model_configs.TransformerModel.d_model,
            nhead=cfg.model_configs.TransformerModel.nhead,
            num_layers=cfg.model_configs.TransformerModel.num_layers,
            dropout=cfg.model_configs.TransformerModel.dropout
        )
    elif model_name == "TemporalFusionTransformer":
        return TemporalFusionTransformer(
            input_size=input_size,
            output_size=output_size,
            output_length=output_length,
            num_layers=cfg.model_configs.TemporalFusionTransformer.num_layers,
            d_model=cfg.model_configs.TemporalFusionTransformer.d_model,
            num_heads=cfg.model_configs.TemporalFusionTransformer.num_heads,
            dropout=cfg.model_configs.TemporalFusionTransformer.dropout
        )
    elif model_name == "ImprovedLSTM":   
        return ImprovedLSTM(
            input_size=input_size,
            hidden_layer_size=cfg.model_configs.ImprovedLSTM.hidden_layer_size,
            output_size=output_size,
            output_length=output_length,
            num_layers=cfg.model_configs.ImprovedLSTM.num_layers,
            dropout=cfg.model_configs.ImprovedLSTM.dropout
        )
    elif model_name == "BiLSTM":    
        return BiLSTM(
            input_size=input_size,
            hidden_layer_size=cfg.model_configs.BiLSTM.hidden_layer_size,
            output_size=output_size,
            output_length=output_length,
            num_layers=cfg.model_configs.BiLSTM.num_layers,
            dropout=cfg.model_configs.BiLSTM.dropout
        )
    else:
        raise ValueError(f"Model {model_name} not supported")