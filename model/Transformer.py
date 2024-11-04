import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, output_length, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_length = output_length
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_projection = nn.Linear(d_model, output_size * output_length)

    def forward(self, src):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_projection(output[:, -1, :])
        return output.view(-1, self.output_length, self.output_size)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(GatedLinearUnit, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.glu = GatedLinearUnit(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        gated_linear_output = self.glu(residual)
        x = x + gated_linear_output
        x = self.layer_norm(x)
        return x

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, output_size, output_length, num_layers, d_model, num_heads, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_length = output_length
        self.d_model = d_model

        self.input_embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
       
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model * 4, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.gated_residual_network = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.output_layer = nn.Linear(d_model, output_size * output_length)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.gated_residual_network(x[:, -1, :])  # Use only the last time step
        x = self.output_layer(x)
        return x.view(-1, self.output_length, self.output_size)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)