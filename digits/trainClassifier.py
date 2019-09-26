import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from digits.dataLoader import spokenDigitDataset
import yaml

with open('/home/ubuntu/projects/digits/digits/classifierParams.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

args = args['classifier']
d_args = args['data']
t_args = args['training']

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

class Slider():
    def __init__(self, size, step):
        self.size = size
        self.step = step
    def __call__(self, inp):
        inp= inp.squeeze().unfold(0, self.size, self.step)
        return inp

tsfm = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    Slider(8, 4)
])
# tsfm = None

trainSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                                         DATA_TYPE,
                                         transform=tsfm,
                                         train = True)

testSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                              DATA_TYPE,
                              transform=tsfm,
                              train=False)

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS)

batch = next(iter(trainLoader))

INPUT_DIM = 512
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
        
        tag_scores = model(batch['feature'].view(BATCH_SIZE, 15, -1))
        loss = criterion(tag_scores[:,-1,:], batch['label'].detach())
        loss.backward()
        optimizer.step()
        running_loss += loss / len(trainSet)
    with torch.no_grad():
        inputs = batch['feature']
        model.eval()
        tag_scores = model(inputs.view(BATCH_SIZE, 15, -1))
        # l = tag_scores[-1,-1,:].argmax()
        acc = (tag_scores[:, -1, :].argmax(1) == batch['label']).sum().item()/BATCH_SIZE
        acc = round(acc, 2)
        print('Loss: ', running_loss.item())
        # print('Target: ', batch['label'][-1])
        # print('Estimate: ', l)
        print('Accuracy: %', acc)
