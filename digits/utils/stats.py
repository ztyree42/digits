import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np

from digits.transforms.mixer import Mixer
from digits.transforms.stft import ToSTFT
from digits.dataLoader import spokenDigitDataset, ToTensor, Latent
from torch.utils.tensorboard import SummaryWriter
import yaml

with open('/home/ubuntu/projects/digits/digits/params.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

args = args['full_decomposer']
d_args = args['data']
f_args = args['transforms']
t_args = args['training']

EMBEDDING_PATH = f_args['embedding_path']
STEP_SIZE = f_args['step_size']
EMBEDDING_HIDDEN = f_args['embedding_hidden']
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
    ToTensor(STEP_SIZE, True),
    Latent(EMBEDDING_PATH,
           hidden_dims=EMBEDDING_HIDDEN, input_dim=EMBEDDING_INPUT,
           full=True)
])


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


trainSet = spokenDigitDataset(DATA_PATH,
                              DATA_TYPE,
                              transform=tsfm,
                              train=True,
                              mixing=MIXING,
                              full=FULL)

testSet = spokenDigitDataset(DATA_PATH,
                             DATA_TYPE,
                             transform=tsfm,
                             train=False,
                             mixing=MIXING,
                             full=FULL)

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS,
                         worker_init_fn=worker_init_fn)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS,
                        worker_init_fn=worker_init_fn)

MEAN = [b['feature'].mean() for i, b in enumerate(trainLoader)]
MEAN = [m.item() for m in MEAN]
STD = [b['feature'].std() for i, b in enumerate(trainLoader)]
STD = [s.item() for s in STD]

#mean 19.5
#std 1869

MEAN2 = [b['label'].mean() for i, b in enumerate(trainLoader)]
MEAN2 = [m.item() for m in MEAN2]
STD2 = [b['label'].std() for i, b in enumerate(trainLoader)]
STD2 = [s.item() for s in STD2]

#mean 21.5
#std 2400

#norm mean 20
#norm std 2130
