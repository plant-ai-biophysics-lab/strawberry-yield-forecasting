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