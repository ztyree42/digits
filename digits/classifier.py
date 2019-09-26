import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_TAGGER(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size):
        super(LSTM_TAGGER, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
            batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, column):
        lstm_out, _ = self.lstm(column)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores
