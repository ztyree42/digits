import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
from transforms.mixer import Mixer
from transforms.stft import ToSTFT
from transforms.normalize import Normalize


from torch.utils.data import DataLoader, Dataset
from features.dataLoader_stft import spokenDigitDataset, ToTensor

STEP_SIZE = 8
BATCH_SIZE = 4
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
                             train=False)

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4)

##
next(iter(trainLoader)).size()
##
class AE(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(AE, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(True),
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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr = learning_rate, weight_decay=1e-5
)

for epoch in range(num_epochs):
    running_loss = 0
    for idx, batch in enumerate(trainLoader):
        # batch = (batch[:, 0, :, :]**2 + batch[:, 1, :, :]**2)**.5
        batch = batch.view(batch.size(0), -1)
        batch.requires_grad_()

        output = model(batch)
        loss = criterion(output, batch.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss / len(trainSet)
        # print('Loss: ', running_loss.item())
        if idx % (len(trainSet) // BATCH_SIZE) == 449:
            print('Loss: ', running_loss.item());

# with torch.no_grad():
#     real = batch.view(batch.size(0), 2, 65, 8).detach().numpy()[0]
#     real = (real[0,:,:]**2 + real[1,:,:]**2)**.5
#     reconstructed = output.view(output.size(0), 2, 65, 8).detach().numpy()[0]
#     reconstructed = (reconstructed[0, :, :]**2 + reconstructed[1, :, :]**2)**.5
#     np.expand_dims(real, 0)
#     np.expand_dims(reconstructed, 0)
#     imgs = [real, reconstructed]
#     plt.imshow(real)

torch.save(model.state_dict(), 'features/models/decomposition/latest.pt')

# loading
# model = AE(64*STEP_SIZE, [128, 64, 32, 16])
# model.load_state_dict(torch.load('features/models/latest.pt'))
# model.eval()

# i = 0
# for idx, batch in enumerate(testLoader):
#     with torch.no_grad():
#         batch = batch.view(batch.size(0), -1)
#         output = model(batch)
#         real = batch.view(batch.size(0), 64, 8).detach().numpy()[0]
#         reconstructed = output.view(output.size(0), 64, 8).detach().numpy()[0]
#         plt.imshow(reconstructed, cmap='gray')
#         i += 1
#         if i == 4:
#             break

# batch = next(iter(trainLoader))
# batch = batch.view(batch.size(0), -1)
# model.encoder(batch[0])

# for i in range(trainSet.__len__()):
#     print(trainSet[i].min(), trainSet[i].max())