
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImprovedLSTM(nn.Module):
        def __init__(self, input_size, hidden_layer_size=100, output_size=10, output_length=13, num_layers=3, dropout=0.2):
            super(ImprovedLSTM, self).__init__()
            self.hidden_layer_size = hidden_layer_size
            self.num_layers = num_layers
            self.output_length = output_length
            self.output_size = output_size
            
            self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True , dropout=dropout)
            self.linear = nn.Linear(hidden_layer_size, output_size * output_length)

        def forward(self, input_seq):
            batch_size = input_seq.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(input_seq.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(input_seq.device)
            
            lstm_out, _ = self.lstm(input_seq, (h0, c0))
            lstm_last = lstm_out[:, -1, :]  # Take the last time step
            linear_out = self.linear(lstm_last)
            
            # Reshape to [batch_size, output_length, output_size]
            predictions = linear_out.view(batch_size, self.output_length, self.output_size)
            
            return predictions



class ModifiedLSTM(nn.Module):

    def __init__(self, original_model, new_input_size, hidden_layer_size=100, output_size=10, output_length=13, num_layers=1):
        super(ModifiedLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.output_length = output_length

        # Create a new input layer to accommodate the new input size
        self.input_layer = nn.Linear(new_input_size, original_model.lstm.input_size)
        
        # Use the LSTM from the original model
        self.lstm = original_model.lstm
        
        # Create a new output layer
        self.fc = nn.Linear(hidden_layer_size, output_size * output_length)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Project the input to the original input size
        x = self.input_layer(x)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Use only the last output
        out = out[:, -1, :]
        
        # Decode the output
        predictions = self.fc(out)
        return predictions.view(batch_size, self.output_length, self.output_size)
    
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=100, output_size=10, output_length=13, num_layers=3, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.output_length = output_length
        self.output_size = output_size
        
        self.bilstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size * output_length)  # *2 because of bidirectional

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_layer_size).to(input_seq.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_layer_size).to(input_seq.device)  # *2 for bidirectional
        
        lstm_out, _ = self.bilstm(input_seq, (h0, c0))
        lstm_last = lstm_out[:, -1, :]  # Take the last time step
        linear_out = self.linear(lstm_last)
        
        # Reshape to [batch_size, output_length, output_size]
        predictions = linear_out.view(batch_size, self.output_length, self.output_size)
        
        return predictions


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_length, num_layers=1, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.output_length = output_length

        # Attention layer
        num_heads = input_size // 2
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size * output_length)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Apply attention
        attn_output, _ = self.attention(x, x, x)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(attn_output, (h0, c0))
        
        # Use only the last output
        out = lstm_out[:, -1, :]
        
        # Decode the output
        predictions = self.fc(out)
        return predictions.view(batch_size, self.output_length, self.output_size)


class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_length, num_layers=1, kernel_size=3, dropout=0.2):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.output_length = output_length

        # CNN layer
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size * output_length)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # CNN forward pass
        cnn_out = self.conv1(x.transpose(1, 2))
        cnn_out = self.relu(cnn_out)
        cnn_out = cnn_out.transpose(1, 2)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_out, (h0, c0))
        
        # Use only the last output
        out = lstm_out[:, -1, :]
        
        # Decode the output
        predictions = self.fc(out)
        return predictions.view(batch_size, self.output_length, self.output_size)
    


    