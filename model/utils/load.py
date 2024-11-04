
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
    else:
        raise ValueError(f"Model {model_name} not supported")