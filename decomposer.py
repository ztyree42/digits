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
    Latent('features/models/decomposition/latest.pt', 
        hidden_dims=[256,128,64,32], input_dim=2*65),
    OneHot(10)
])

trainSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                              'recordings',
                              transform=tsfm,
                              train=True,
                              mixing = True)

testSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                             'recordings',
                             transform=tsfm,
                             train=False,
                             mixing=True)

BATCH_SIZE = 60     

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, worker_init_fn=worker_init_fn)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, worker_init_fn=worker_init_fn)


INPUT_DIM = 32
HIDDEN_DIM = 128
OUTPUT_DIM = 10


# class LSTM_TAGGER(nn.Module):

#     def __init__(self, input_dim, hidden_dim, target_size):
#         super(LSTM_TAGGER, self).__init__()
#         self.hidden_dim = hidden_dim

#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)

#         self.label1 = nn.Linear(hidden_dim, target_size)
#         self.label2 = nn.Linear(hidden_dim, target_size)

#     def forward(self, column):
#         lstm_out, _ = self.lstm(column)
#         logits1 = self.label1(lstm_out)
#         score1 = F.log_softmax(logits1, dim=2)
#         logits2 = self.label2(lstm_out)
#         score2 = F.log_softmax(logits2, dim=2)
#         return score1, score2

class LSTM_TAGGER(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size):
        super(LSTM_TAGGER, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, num_layers=2)

        self.label = nn.Linear(hidden_dim, target_size)

        self.norm = nn.LayerNorm(target_size)

    def forward(self, column):
        lstm_out, _ = self.lstm(column)
        logits = self.label(lstm_out)
        normedLogits = self.norm(logits)
        score = torch.sigmoid(normedLogits)
        return score


model = LSTM_TAGGER(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.cuda()
# criterion = nn.NLLLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)

# for epoch in range(1):
#     train_loss = 0
#     np.random.seed()
#     for idx, batch in enumerate(trainLoader):
#         model.zero_grad()
#         batch['feature'].requires_grad_()

#         score1, score2 = model(batch['feature'])
        
#         loss1 = criterion(score1[:, -1, :], batch['label'][:,0].detach())
#         loss2 = criterion(score2[:, -1, :], batch['label'][:,1].detach())
        
#         loss = loss1 + loss2
#         loss.backward()
#         optimizer.step()
#         train_loss += loss / len(trainSet)
#         if idx % (len(trainSet) // BATCH_SIZE) == 449:
#             with torch.no_grad():
#                 inputs = batch['feature']
#                 score1, score2 = model(inputs)
#                 _, l1 = score1[:, -1, :].max(1)
#                 _, l2 = score2[:, -1, :].max(1)
#                 print('Loss: ', train_loss.item())
#                 print('Target: ', batch['label'])
#                 print('Estimate: ', [l1, l2])

for epoch in range(50):
    train_loss = 0
    val_loss = 0
    np.random.seed()
    for idx, batch in enumerate(trainLoader):
        model.zero_grad()
        batch['feature'].requires_grad_()

        score = model(batch['feature'].cuda())
        loss = criterion(score[:, -1, :], batch['label'].cuda().detach())

        loss.backward()
        optimizer.step()
        train_loss += loss / len(trainSet)
    if (epoch % 5) == 0:
        with torch.no_grad():
            for idx, batch in enumerate(testLoader):
                score = model(batch['feature'].cuda())
                loss = criterion(score[:, -1, :], batch['label'].cuda())
                val_loss += loss / len(testSet)
    writer.add_scalar('loss/val', val_loss, epoch)
    writer.add_scalar('loss/train', train_loss, epoch)
        # if idx % (len(trainSet) // BATCH_SIZE) == 59:
        #     with torch.no_grad():
        #         inputs = batch['feature'].cuda()
        #         score = model(inputs)
        #         _, l = score[:, -1, :].topk(2,1)
        #         v, t = batch['label'].topk(2,1)
        #         print('Loss: ', train_loss.item())
        #         print('Target: ', t)
        #         print('Estimate: ', l)


torch.save(model.state_dict(), 'models/decomposition/latest.pt')
