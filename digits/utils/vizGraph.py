from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models
from torchviz import make_dot
from digits.classifer import LSTM_TAGGER
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from digits.dataLoader import spokenDigitDataset, ToTensor, Latent

tsfm = tv.transforms.Compose([
    ToTensor(8),
    Latent('digits/features/models/classification/latest.pt')
])

trainSet = spokenDigitDataset('/home/ztyree/projects/spokenDigits',
                              'spectrograms',
                              transform=tsfm,
                              train=True)

testSet = spokenDigitDataset('/home/ztyree/projects/spokenDigits',
                             'spectrograms',
                             transform=tsfm,
                             train=False)

BATCH_SIZE = 1

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4)

model = LSTM_TAGGER(16, 64, 10)

# inputs = next(iter(trainLoader))
inputs = torch.randn(1, 3, 224, 224);
model = models.resnet18()
# TODO: Import model from classifier.py and make_dot

y = model.eval()(Variable(inputs).detach())
print(y)

g = make_dot(y)
g
