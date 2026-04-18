"""LSTM model that predicts the clean sinusoid selected by the one-hot input."""
import torch
from torch import nn

import config


class FrequencyExtractorLSTM(nn.Module):
    """(S[t], one_hot_C) → Target_i[t] via stateful LSTM + linear head.

    Input  : (B, L, 5)
    Output : (B, L, 1), (h, c)
    """

    def __init__(
        self,
        input_size: int = config.INPUT_SIZE,
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers: int = config.NUM_LAYERS,
        output_size: int = config.OUTPUT_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def init_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward(self, x: torch.Tensor, state=None):
        out, state = self.lstm(x, state)
        y = self.head(out)
        return y, state
