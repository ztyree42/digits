import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

from transforms.mixer import Mixer
from transforms.stft import ToSTFT
from dataLoader import spokenDigitDataset, ToTensor, Latent
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

tsfm = tv.transforms.Compose([
    Mixer(),
    ToSTFT(),
    ToTensor(8),
    Latent('/home/ubuntu/projects/digits/digits/features/models/decomposition/latest.pt',
           hidden_dims=[256, 128, 64, 32], input_dim=2*65)
])

trainSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                              'recordings',
                              transform=tsfm,
                              train=True,
                              mixing=True,
                              full=True)

testSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                             'recordings',
                             transform=tsfm,
                             train=False,
                             mixing=True,
                             full=True)

BATCH_SIZE = 60


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=8, worker_init_fn=worker_init_fn)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=8, worker_init_fn=worker_init_fn)


INPUT_DIM = 32
HIDDEN_DIM = 256
OUTPUT_DIM = 10


class LSTM_DECOMPOSER(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size, drop_out=.5):
        super(LSTM_TAGGER, self).__init__()
        self.hidden_dim = hidden_dim
        self.drop_out = nn.Dropout(drop_out)
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2,
                            batch_first=True, num_layers=2
                            bidirectional=True)

        self.label = nn.Linear(hidden_dim, target_size)

        self.norm = nn.LayerNorm(target_size)

    def forward(self, column):
        column = self.drop_out(column)
        lstm_out, _ = self.lstm(column)
        logits = self.label(lstm_out)
        # normedLogits = self.norm(logits)
        # score = torch.sigmoid(normedLogits)
        return logits
