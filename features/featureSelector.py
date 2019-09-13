import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader, Dataset
from features.dataLoader import spokenDigitDataset, ToTensor

STEP_SIZE = 8
BATCH_SIZE = 4
# INPUT_DIM = (BATCH_SIZE, )

trainSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                              'spectrograms',
                              transform=ToTensor(STEP_SIZE),
                              train=True)

testSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                             'spectrograms',
                             transform=ToTensor(STEP_SIZE),
                             train=False)

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4)

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


# num_epochs = 10
# learning_rate = 1e-3

# model = AE(64*STEP_SIZE, [128,64,32,16])

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(
#     model.parameters(), lr = learning_rate, weight_decay=1e-5
# )
        
# for epoch in range(500):
#     running_loss = 0
#     for idx, batch in enumerate(trainLoader):
#         batch = batch.view(batch.size(0), -1)
#         batch.requires_grad_()
        
#         output = model(batch)
#         loss = criterion(output, batch)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss / len(trainSet)

#         if idx % (len(trainSet) // BATCH_SIZE) == 449:
#             print('Loss: ', running_loss.item())

# with torch.no_grad():
#     real = batch.view(batch.size(0), 64, 8).detach().numpy()[0]
#     reconstructed = output.view(output.size(0), 64, 8).detach().numpy()[0]
#     np.expand_dims(real, 0)
#     np.expand_dims(reconstructed, 0)
#     imgs = [real, reconstructed]
#     plt.imshow(real)

#     _, axs = plt.subplots(1, 2, figsize=(1, 2))
#     for img, ax in zip(imgs, axs):
#         ax.imshow(img)
#         plt.show()
# torch.save(model.state_dict(), 'features/models/latest.pt')

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
