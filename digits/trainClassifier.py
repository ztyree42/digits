import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from digits.transforms.slider import Slider
from digits.dataLoader import spokenDigitDataset
import yaml
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

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

tsfm = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    Slider(8, 4)
])
# tsfm = None

trainSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                              DATA_TYPE,
                              transform=tsfm,
                              train=True)

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

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, column):
        lstm_out, _ = self.lstm(column)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores


model = LSTM_TAGGER(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)

best_loss = None
for epoch in range(100):
    train_loss = 0
    val_loss = 0
    for idx, batch in enumerate(trainLoader):
        model.zero_grad()
        model.train()
        batch['feature'].requires_grad_()

        tag_scores = model(batch['feature'].cuda().view(BATCH_SIZE, 15, -1))
        loss = criterion(tag_scores[:, -1, :], batch['label'].cuda().detach())
        loss.backward()
        optimizer.step()
        train_loss += BATCH_SIZE*(loss / len(trainSet))
    with torch.no_grad():
        for idx, batch in enumerate(testLoader):
            batch = next(iter(testLoader))
            inputs = batch['feature'].cuda()
            tag_scores = model.eval()(inputs.view(BATCH_SIZE, 15, -1))
            loss = criterion(tag_scores[:, -1, :],
                             batch['label'].cuda().detach())
            val_loss += BATCH_SIZE*(loss/len(testSet))
        writer.add_scalar('classifier/loss/val', val_loss, epoch)
        acc = (tag_scores[:, -1, :].argmax(1) ==
               batch['label'].cuda()).sum().item()/BATCH_SIZE
        acc = round(acc, 2)
        print('Loss: ', val_loss.item())
        print('Accuracy: %', acc)
        if best_loss is None:
            best_loss = val_loss
        if val_loss < best_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            best_loss = val_loss
    writer.add_scalar('classifier/loss/train', train_loss, epoch)
