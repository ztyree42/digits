import torch
import torch.nn as nn

class AE(nn.Module):
    """Denoising Auto Encoder."""
    def __init__(self, input_dim, hidden_dim, dropout=.5):
        super(AE, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(True),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(True),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(True),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim[2], hidden_dim[3])
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim[3], hidden_dim[2]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[2], hidden_dim[1]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[0], input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
