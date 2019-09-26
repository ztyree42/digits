from digits.features.dataLoader_stft import spokenDigitDataset, ToTensor
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from digits.transforms.mixer import Mixer
from digits.transforms.stft import ToSTFT
from digits.transforms.normalize import Normalize
# import argparse

# parser = argparse.ArgumentParser('train denoise ae')
# parser.add_argument('--fine-tune', dest='ft', const=True,
#     nargs='?', default=False)
# args = parser.parse_args()

writer = SummaryWriter()

STEP_SIZE = 8
BATCH_SIZE = 50
# INPUT_DIM = (BATCH_SIZE, )

tsfm = tv.transforms.Compose([
    Mixer(),
    ToSTFT(),
    ToTensor(STEP_SIZE),
    Normalize(44.5,151,44.5,151,False)
])
# tsfm = tv.transforms.Compose([
#     Mixer(),
#     ToSTFT()
# ])

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


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, worker_init_fn=worker_init_fn)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, worker_init_fn=worker_init_fn)

class AE(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=.9):
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
            # nn.ReLU(True),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim[3], hidden_dim[4])
            # nn.ReLU(True),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim[4], hidden_dim[5])
        )
        self.decoder = nn.Sequential(
            # nn.Linear(hidden_dim[5], hidden_dim[4]),
            # nn.ReLU(True),
            # nn.Linear(hidden_dim[4], hidden_dim[3]),
            # nn.ReLU(True),
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


# class ConvDAE(nn.Module):
#     def __init__(self):
#         super(ConvDAE, self).__init__()

#         # input: batch x 3 x 32 x 32 -> output: batch x 16 x 16 x 16
#         self.encoder = nn.Sequential(
#             nn.Conv2d(2, 16, 3, stride=1, padding=1),  # batch x 16 x 32 x 32
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.MaxPool2d(2, stride=2, return_indices=True,
#                 padding=0)
#         )

#         self.unpool = nn.MaxUnpool2d(2, stride=2)

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(16, 16, 3, stride=1,
#                                padding=1, output_padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.ConvTranspose2d(16, 2, 3, stride=1,
#                                padding=1, output_padding=0),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         size = x.size()
#         x = x.view(-1, 2, 65, 8)
#         out, indices = self.encoder(x)
#         out = self.unpool(out, indices, output_size=x.size())
#         out = self.decoder(out)
#         out = out.view(size)
#         return out

num_epochs = 500
learning_rate = 1e-3

model = AE(65*STEP_SIZE, [512, 256, 128, 128])

# model = ConvDAE()
# if args.ft:
#     model.load_state_dict(torch.load(
#         '/home/ubuntu/projects/digits/digits/features/models/decomposition/latest.pt'))
model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
model.parameters(), lr=learning_rate, weight_decay=0.
)

best_loss = None
for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0
    np.random.seed()
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
                loss = criterion(output, batch.cuda().detach())
                val_loss += loss / len(testSet)
        writer.add_scalar('dae/loss/val', val_loss, epoch)
        if best_loss is None:
            best_loss = val_loss
        if val_loss < best_loss:
            torch.save(model.state_dict(),
                        '/home/ubuntu/projects/digits/digits/features/models/decomposition/latest.pt')
            best_loss = val_loss
    writer.add_scalar('dae/loss/train', train_loss, epoch)
