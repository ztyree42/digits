import torch
import torch.nn as nn

class LSTM_DECOMPOSER(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size, drop_out=.5):
        super(LSTM_DECOMPOSER, self).__init__()
        self.hidden_dim = hidden_dim
        # self.drop_out = nn.Dropout(drop_out)
        # self.lstm = nn.LSTM(input_dim, hidden_dim // 2,
        #                     batch_first=True, num_layers=2,
        #                     bidirectional=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim//2,
                            batch_first=True, num_layers=5,
                            bidirectional=True)
        self.linear11 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear21 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear12 = nn.Linear(hidden_dim//2, target_size)
        self.linear22 = nn.Linear(hidden_dim//2, target_size)
        self.relu = nn.ReLU()

    def forward(self, column):
        # column = self.drop_out(column)
        lstm_out, _ = self.lstm(column)
        out1 = self.linear11(lstm_out)
        out1 = self.relu(out1)
        out1 = self.linear12(out1)
        out2 = self.linear21(lstm_out)
        out2 = self.relu(out2)
        out2 = self.linear22(out2)
        return torch.stack([out1, out2], 1)
