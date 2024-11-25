import torch
from torch import nn

class LSTMModel(nn.Module):
    
    def __init__(self, input_dim):
        
        super(LSTMModel, self).__init__()
        
        # lstm layers
        self.lstm = nn.Sequential(
            nn.LSTM(input_dim, 80, batch_first=True),
            nn.LSTM(80, 40, batch_first=True),
            nn.LSTM(40, 40, batch_first=True),
            nn.LSTM(40, 20, batch_first=True)
        )
        
        # fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        
        # pass through lstm layers
        for lstm_layer in self.lstm:
            x, _ = lstm_layer(x)
            
        # get last hidden state (sequence-to-one)
        x = x[:, -1, :]
        
        # fc layers
        x = self.fc_layers(x)
        
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, num_heads=2, ff_dim=128, num_layers=2, max_seq_len=4):
        super(TransformerDecoder, self).__init__()
        
        # Positional embeddings
        self.positional_embeddings = nn.Embedding(max_seq_len, ff_dim)
        
        # Decoder layers
        self.embedding = nn.Linear(input_dim, ff_dim)  # Project input to model dimension
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=ff_dim, nhead=num_heads, dim_feedforward=ff_dim
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(ff_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
    def forward(self, x):
        # Embed the input
        x = self.embedding(x)
        
        # Add positional embeddings
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], seq_len)
        pos_emb = self.positional_embeddings(positions)
        x = x + pos_emb
        
        # Permute for transformer compatibility (seq_len, batch_size, ff_dim)
        x = x.permute(1, 0, 2)
        
        # Prepare memory
        memory = torch.zeros_like(x)
        x = self.transformer_decoder(x, memory)
        
        # Get the last hidden state (sequence-to-one)
        x = x[-1, :, :]
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x
