import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

from transforms.mixer import Mixer
from transforms.stft import ToSTFT
from dataLoader import spokenDigitDataset, ToTensor, Latent
from torch.utils.tensorboard import SummaryWriter
import yaml

with open('/home/ztyree/projects/digits/digits/params.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

args = args['full_decomposer']
d_args = args['data']
f_args = args['transforms']
t_args = args['training']

EMBEDDING_PATH = f_args['embedding_path']
STEP_SIZE = f_args['step_size']
EMEDDING_HIDDEN = f_args['embedding_hidden']
EMBEDDING_INPUT = f_args['embedding_input']

DATA_PATH = d_args['path']
DATA_TYPE = d_args['type']
MIXING = d_args['mixing']
FULL = d_args['full']

BATCH_SIZE = t_args['batch_size']
NUM_WORKERS = t_args['num_workers']
INPUT_DIM = t_args['input_dim']
HIDDEN_DIM = t_args['hidden_dim']
OUTPUT_DIM = t_args['output_dim']
LEARNING_RATE = t_args['learning_rate']
WEIGHT_DECAY = t_args['weight_decay']
EPOCHS = t_args['epochs']
DROP_OUT = t_args['drop_out']
MODEL_PATH = t_args['model_path']

writer = SummaryWriter()

tsfm = tv.transforms.Compose([
    Mixer(),
    ToSTFT(),
    ToTensor(STEP_SIZE),
    Latent(EMBEDDING_PATH,
           hidden_dims=EMEDDING_HIDDEN, input_dim=EMBEDDING_INPUT),
    OneHot(10)
])


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


trainSet = spokenDigitDataset(DATA_PATH,
                              DATA_TYPE,
                              transform=tsfm,
                              train=True,
                              mixing=MIXING,
                              full = FULL)

testSet = spokenDigitDataset(DATA_PATH,
                             DATA_TYPE,
                             transform=tsfm,
                             train=False,
                             mixing=MIXING,
                             full = FULL)

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS,
                         worker_init_fn=worker_init_fn)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS,
                        worker_init_fn=worker_init_fn)


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
