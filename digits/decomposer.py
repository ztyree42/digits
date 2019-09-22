import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transforms.mixer import Mixer
from transforms.stft import ToSTFT
from transforms.normalize import Normalize
from transforms.oneHot import OneHot
import torchvision as tv
from dataLoader import spokenDigitDataset, ToTensor, Latent
from torch.utils.tensorboard import SummaryWriter
import yaml

with open('/home/ubuntu/projects/digits/digits/params.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

args = args['decomposer']
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

BATCH_SIZE = t_args['batch_size']
NUM_WORKERS =t_args['num_workers']
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
        hidden_dims=EMBEDDING_HIDDEN, input_dim=EMBEDDING_INPUT),
    OneHot(10)
])

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

trainSet = spokenDigitDataset(DATA_PATH,
                              DATA_TYPE,
                              transform=tsfm,
                              train=True,
                              mixing=MIXING)

testSet = spokenDigitDataset(DATA_PATH,
                             DATA_TYPE,
                             transform=tsfm,
                             train=False,
                             mixing=MIXING)     

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, 
                         worker_init_fn=worker_init_fn)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, 
                        worker_init_fn=worker_init_fn)

class LSTM_TAGGER(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size, drop_out=DROP_OUT):
        super(LSTM_TAGGER, self).__init__()
        self.hidden_dim = hidden_dim
        self.drop_out = nn.Dropout(drop_out)
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2,
                            batch_first=True, num_layers=2,
                            bidirectional=True)

        self.dense_0 = nn.Linear(hidden_dim, hidden_dim//2)
        self.dense_1 = nn.Linear(hidden_dim//2, target_size)
        self.relu = nn.ReLU()

        self.norm = nn.LayerNorm(target_size)

    def forward(self, column):
        column = self.drop_out(column)
        lstm_out, _ = self.lstm(column)
        logits = self.dense_0(lstm_out)
        logits = self.relu(logits)
        logits = self.dense_1(logits)
        # normedLogits = self.norm(logits)
        # score = torch.sigmoid(normedLogits)
        return logits

model = LSTM_TAGGER(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), 
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

best_loss = None
for epoch in range(EPOCHS):
    train_loss = 0
    val_loss = 0
    np.random.seed()
    for idx, batch in enumerate(trainLoader):
        model.zero_grad()
        model.train()
        batch['feature'].requires_grad_()

        score = model(batch['feature'].cuda())
        loss = criterion(score[:, -1, :], batch['label'].cuda().detach())

        loss.backward()
        optimizer.step()
        train_loss += BATCH_SIZE*(loss / len(trainSet))
    if (epoch % 5) == 0:
        with torch.no_grad():
            for idx, batch in enumerate(testLoader):
                score = model.eval()(batch['feature'].cuda())
                loss = criterion(score[:, -1, :], batch['label'].cuda())
                val_loss += BATCH_SIZE*(loss / len(testSet))
        writer.add_scalar('mixClass/loss/val', val_loss, epoch)
        if best_loss is None:
            best_loss = val_loss
        if val_loss < best_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            best_loss = val_loss
    writer.add_scalar('mixClass/loss/train', train_loss, epoch)
