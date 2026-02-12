import torch.nn as nn

"""Baseline LSTM Model"""


class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [B, T, H]
        return self.fc(lstm_out)  # [B, T, O]
