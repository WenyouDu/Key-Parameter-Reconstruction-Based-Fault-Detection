import torch.nn as nn

"""Add input feature extraction layer"""


class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size

        # ================= Add input feature extraction layer =================
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Xavier initialization to help the model converge quickly"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:  # Only perform Xavier initialization on matrices
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                # Initialize forget gate bias to 1 (helps gradient flow, prevents vanishing gradient)
                if 'lstm.bias_ih' in name or 'lstm.bias_hh' in name:
                    param.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # 1. Input feature extraction (Input -> Linear -> ReLU)
        x = self.input_fc(x)
        x = self.relu(x)

        # 2. Sequential feature extraction (LSTM)
        lstm_out, _ = self.lstm(x)

        # 3. Output mapping (Linear)
        out = self.fc(lstm_out)

        return out

