import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from dataLoader import spokenDigitDataset, ToTensor, Latent

tsfm = tv.transforms.Compose([
    ToTensor(8),
    Latent('features/models/classification/latest.pt')
])

trainSet = spokenDigitDataset('/home/ztyree/projects/spokenDigits',
                                         'spectrograms',
                                         transform=tsfm,
                                         train = True)

testSet = spokenDigitDataset('/home/ztyree/projects/spokenDigits',
                              'spectrograms',
                              transform=tsfm,
                              train=False)

BATCH_SIZE = 4

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4)


INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 10


class LSTM_TAGGER(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size):
        super(LSTM_TAGGER, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, column):
        lstm_out, _ = self.lstm(column)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores


model = LSTM_TAGGER(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)

for epoch in range(20):
    running_loss = 0
    for idx, batch in enumerate(trainLoader):
        model.zero_grad()
        batch['feature'].requires_grad_()
        
        tag_scores = model(batch['feature'])
        loss = criterion(tag_scores[:,-1,:], batch['label'])
        loss.backward()
        optimizer.step()
        running_loss += loss / len(trainSet)
        if idx % (len(trainSet) // BATCH_SIZE) == 449:
            with torch.no_grad():
                inputs = batch['feature']
                tag_scores = model(inputs)
                _, l = tag_scores[:,-1,:].max(1)
                print('Loss: ', running_loss.item())
                print('Target: ', batch['label'])
                print('Estimate: ', l)
