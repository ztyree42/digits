import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from digits.transforms.slider import Slider
from digits.mixedDataLoader import MixedDigits
import yaml
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

with open('/home/ubuntu/projects/digits/digits/mixedClassifierParams.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

args = args['mixedClassifier']
d_args = args['data']
t_args = args['training']

DATA_PATH = d_args['path']
DATA_TYPE = d_args['type']

BATCH_SIZE = t_args['batch_size']
NUM_WORKERS = t_args['num_workers']
INPUT_DIM = t_args['input_dim']
HIDDEN_DIM = t_args['hidden_dim']
OUTPUT_DIM = t_args['output_dim']
NUM_LAYERS = t_args['num_layers']
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

trainSet = MixedDigits('/home/ubuntu/projects/spokenDigits',
                              DATA_TYPE,
                              transform=tsfm,
                              train=True)

testSet = MixedDigits('/home/ubuntu/projects/spokenDigits',
                             DATA_TYPE,
                             transform=tsfm,
                             train=False)

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, drop_last=True)

testLoader = DataLoader(testSet, batch_size=len(testSet), shuffle=False,
                        num_workers=NUM_WORKERS, drop_last=True)

batch = next(iter(trainLoader))

INPUT_DIM = 512
HIDDEN_DIM = 64
OUTPUT_DIM = 10


class LSTM_TAGGER(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size):
        super(LSTM_TAGGER, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=NUM_LAYERS,
                            batch_first=True)

        self.dense1 = nn.Linear(hidden_dim, target_size)
        self.dense2 = nn.Linear(hidden_dim, target_size)

    def forward(self, column):
        lstm_out, _ = self.lstm(column)
        out1 = self.dense1(lstm_out)
        out1 = F.log_softmax(out1, dim=2)
        out2 = self.dense2(lstm_out)
        out2 = F.log_softmax(out2, dim=2)
        return out1, out2


model = LSTM_TAGGER(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)


best_loss = None
for epoch in range(1):
    train_loss = 0
    val_loss = 0
    for idx, batch in enumerate(trainLoader):
        model.zero_grad()
        model.train()
        batch['feature'].requires_grad_()

        out1, out2 = model(batch['feature'].cuda().view(BATCH_SIZE, 15, -1))
        loss1 = criterion(out1[:, -1, :], batch['label'][0].cuda().detach())
        loss2 = criterion(out2[:, -1, :], batch['label'][1].cuda().detach())
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        train_loss += BATCH_SIZE*(loss / len(trainSet))
        print("torch.cuda.memory_allocated: %fGB" %
              (torch.cuda.memory_allocated()/1024/1024/1024))
        print("torch.cuda.memory_cached: %fGB" %
              (torch.cuda.memory_cached()/1024/1024/1024))
    with torch.no_grad():
        for idx, batch in enumerate(testLoader):
            batch = next(iter(testLoader))
            inputs = batch['feature'].cuda()
            out1, out2 = model.eval()(inputs.view(BATCH_SIZE, 15, -1))
            loss1 = criterion(
                out1[:, -1, :], batch['label'][0].cuda().detach())
            loss2 = criterion(
                out2[:, -1, :], batch['label'][1].cuda().detach())
            loss = loss1 + loss2
            val_loss += loss
        writer.add_scalar('classifier/loss/val', val_loss, epoch)
        acc1 = (out1[:, -1, :].argmax(1) ==
               batch['label'][0].cuda()).sum().item()/len(testSet)
        acc2 = (out2[:, -1, :].argmax(1) ==
               batch['label'][1].cuda()).sum().item()/len(testSet)
        acc1 = round(acc1, 2)
        acc2 = round(acc2, 2)
        print('Loss: ', val_loss.item())
        print('Accuracy1: %', acc1)
        print('Accuracy2: %', acc2)
        if best_loss is None:
            best_loss = val_loss
        if val_loss < best_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            best_loss = val_loss
    writer.add_scalar('classifier/loss/train', train_loss, epoch)
