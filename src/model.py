import torch
import math

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
            nn.Linear(20, 1),
            nn.ReLU() # remove negative values from output
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
    
class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings module.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):

        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and causal masking."""
    def __init__(self, hidden_size=128, num_heads=4):
        super(TransformerBlock, self).__init__()
        
        # Layer norm for input
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.1
        )
        
        # Layer norm after attention
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, x, padding_mask=None, attn_mask=None):
        # Layer norm
        norm_x = self.norm1(x)
        
        # Multi-head attention with residual connection
        x = self.multihead_attn(
            norm_x, norm_x, norm_x, attn_mask=attn_mask, key_padding_mask=padding_mask
        )[0] + x
        
        # Layer norm
        norm_x = self.norm2(x)
        
        # Feedforward with residual connection
        x = self.mlp(norm_x) + x
        return x
    
class Transformer(nn.Module):
    """Decoder-only Transformer for time-series forecasting with causal masking."""
    def __init__(self, input_dim, hidden_size=128, num_layers=3, num_heads=4):
        super(Transformer, self).__init__()
        
        # Project input features to hidden size
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # Positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, 1)
        
    def forward(self, input_seq, causal_mask=None):
        # Embeddings and positional encodings
        input_embs = self.input_projection(input_seq)
        seq_len = input_seq.size(1)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=input_seq.device)).unsqueeze(0)
        embs = input_embs + pos_emb

        # Pass through Transformer blocks
        for block in self.blocks:
            embs = block(embs, padding_mask=None, attn_mask=causal_mask)

        # Get last timepoint's representation
        final_hidden = embs[:, -1, :]
        return self.fc_out(final_hidden)

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, num_heads=2, ff_dim=128, num_layers=2):
        super(TransformerDecoder, self).__init__()

        # Sinusoidal positional embeddings
        self.positional_embeddings = SinusoidalPosEmb(ff_dim)

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
        positions = torch.arange(seq_len, device=x.device).float()
        pos_emb = self.positional_embeddings(positions).unsqueeze(0).expand(x.shape[0], -1, -1)
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
    
class LSTMTransformer(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim=128, lstm_layers=2, 
                 num_heads=4, ff_dim=256, transformer_layers=3, max_seq_len=10, dropout=0.1):
        super(LSTMTransformer, self).__init__()
        
        # LSTM Encoder
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Transformer Decoder
        self.embedding = nn.Linear(2 * lstm_hidden_dim, ff_dim)  # BiLSTM doubles hidden_dim
        self.positional_embeddings = nn.Embedding(max_seq_len, ff_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=ff_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)
        
        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(ff_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Single output (regression)
        )
    
    def forward(self, x):
        # LSTM Encoder
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, 2 * lstm_hidden_dim)
        
        # Transformer Decoder
        x = self.embedding(lstm_out)  # Project LSTM output to transformer input size
        seq_len = x.shape[1]
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], seq_len)
        x = x + self.positional_embeddings(positions)
        
        # Permute for transformer compatibility
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, ff_dim)
        memory = torch.zeros_like(x)  # Placeholder memory
        
        x = self.transformer_decoder(x, memory)
        
        # Sequence-to-one prediction: Use last hidden state
        x = x[-1, :, :]  # Shape: (batch_size, ff_dim)
        
        # Fully Connected Layers
        x = self.fc_layers(x)
        
        return x
    
class LSTMTransformerMasked(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, transformer_hidden_size, num_transformer_layers, num_heads, max_seq_len=10, dropout=0.1, use_time_emb=True):
        super(LSTMTransformerMasked, self).__init__()
        
        self.use_time_emb = use_time_emb
        
        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden_dim, num_layers=lstm_layers, 
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Transformer
        self.input_projection = nn.Linear(lstm_hidden_dim * 2, transformer_hidden_size)  # Bidirectional LSTM doubles the hidden size
        self.pos_emb = nn.Embedding(max_seq_len, transformer_hidden_size)
        self.time_emb = nn.Sequential(
            nn.Linear(1, transformer_hidden_size),
            nn.ReLU(),
            nn.Linear(transformer_hidden_size, transformer_hidden_size)
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_hidden_size, num_heads) for _ in range(num_transformer_layers)
        ])
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(transformer_hidden_size, transformer_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(transformer_hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # LSTM Encoder
        lstm_out, _ = self.lstm(x)  # Output shape: (batch_size, seq_len, 2 * lstm_hidden_dim)
        
        # Project LSTM output to Transformer input size
        x = self.input_projection(lstm_out)
        
        # Add positional embeddings and time embeddings
        seq_len = x.size(1)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device)).unsqueeze(0)
        if self.use_time_emb:
            time = x[:, :, -1:]
            time_emb = self.time_emb(time)
            x = x + pos_emb + time_emb
        else:
            x = x + pos_emb
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Get the last hidden state
        x = x[:, -1, :]
        
        # Fully connected layers
        x = self.fc(x)
        
        return x
    
class TransformerEncoderTransformerMasked(nn.Module):
    def __init__(self, input_dim, transformer_hidden_size, num_encoder_layers,
                 num_transformer_layers, num_heads, max_seq_len=10, dropout=0.1, use_time_emb=True):
        super(TransformerEncoderTransformerMasked, self).__init__()
        
        self.use_time_emb = use_time_emb
        
        # Transformer encoder
        self.input_projection = nn.Linear(input_dim, transformer_hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, transformer_hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        if self.use_time_emb:
            self.time_emb = nn.Sequential(
                nn.Linear(1, transformer_hidden_size),
                nn.ReLU(),
                nn.Linear(transformer_hidden_size, transformer_hidden_size)
            )
        
        # Transformer decoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_hidden_size, num_heads) 
            for _ in range(num_transformer_layers)
        ])
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(transformer_hidden_size, transformer_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(transformer_hidden_size // 2, 1)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Transformer Encoder
        x = self.input_projection(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device)).unsqueeze(0)
        x = x + pos_emb
        if self.use_time_emb:
            time = x[:, :, -1:]  # (batch_size, seq_len, 1)
            time_emb = self.time_emb(time)
            x = x + time_emb
        x = self.transformer_encoder(x)
        
        # Transformer decoder
        for block in self.transformer_blocks:
            x = block(x)
        
        x = x[:, -1, :]
        x = self.fc(x)
        return x