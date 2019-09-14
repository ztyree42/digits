import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from digits.transforms.mixer import Mixer
from digits.transforms.stft import ToSTFT
from digits.transforms.normalize import Normalize


from torch.utils.data import DataLoader, Dataset
from digits.features.dataLoader_stft import spokenDigitDataset, ToTensor

writer = SummaryWriter()

STEP_SIZE = 8
BATCH_SIZE = 60
# INPUT_DIM = (BATCH_SIZE, )

tsfm = tv.transforms.Compose([
    Mixer(),
    ToSTFT(),
    ToTensor(STEP_SIZE)
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

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4)


class AE(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=.5):
        super(AE, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(True),
            nn.Dropout(dropout),
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


num_epochs = 200
learning_rate = 1e-3

model = AE(2*65*STEP_SIZE, [256,128,64,32])
model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr = learning_rate, weight_decay=1e-5
)

best_loss = None
for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0
    for idx, batch in enumerate(trainLoader):
        model.zero_grad()
        model.train()
        batch = batch.cuda().view(batch.size(0), -1)
        batch.requires_grad_()

        output = model(batch.cuda())
        loss = criterion(output, batch.cuda().detach())

        loss.backward()
        optimizer.step()
        train_loss += loss / len(trainSet)
    if (epoch % 5) == 0:
        with torch.no_grad():
            for idx, batch in enumerate(testLoader):
                batch = batch.cuda().view(batch.size(0), -1)
                output = model.eval()(batch.cuda())
                loss = criterion(output, batch.cuda())
                val_loss += loss / len(testSet)
        writer.add_scalar('dae/loss/val', val_loss, epoch)
        if best_loss is None:
            best_loss = val_loss
        if val_loss < best_loss:
            torch.save(model.state_dict(),
                       '/home/ubuntu/projects/digits/digits/features/models/decomposition/latest.pt')
            best_loss = val_loss
    writer.add_scalar('dae/loss/train', train_loss, epoch)
