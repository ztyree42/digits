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

writer = SummaryWriter()

tsfm = tv.transforms.Compose([
    Mixer(),
    ToSTFT(),
    ToTensor(8),
    Latent('/home/ubuntu/projects/digits/digits/features/models/decomposition/latest.pt', 
        hidden_dims=[256,128,64,32], input_dim=2*65),
    OneHot(10)
])

trainSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                              'recordings',
                              transform=tsfm,
                              train=True,
                              mixing=True)

testSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                             'recordings',
                             transform=tsfm,
                             train=False,
                             mixing=True)

BATCH_SIZE = 60     

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=8, worker_init_fn=worker_init_fn)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=8, worker_init_fn=worker_init_fn)


INPUT_DIM = 32
HIDDEN_DIM = 128
OUTPUT_DIM = 10

class LSTM_TAGGER(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size, drop_out=.5):
        super(LSTM_TAGGER, self).__init__()
        self.hidden_dim = hidden_dim
        self.drop_out = nn.Dropout(drop_out)
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, num_layers=2,
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

model = LSTM_TAGGER(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)

best_loss = None
for epoch in range(200):
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
            torch.save(model.state_dict(), '/home/ubuntu/projects/digits/digits/models/decomposition/latest.pt')
            best_loss = val_loss
    writer.add_scalar('mixClass/loss/train', train_loss, epoch)
