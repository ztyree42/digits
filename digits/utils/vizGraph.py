from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models
from torchviz import make_dot
from digits.classifier import LSTM_TAGGER
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from digits.transforms.slider import Slider
from digits.dataLoader import spokenDigitDataset

tsfm = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    Slider(8, 4)
])

trainSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                              'spectrograms',
                              transform=tsfm,
                              train=True)

testSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                             'spectrograms',
                             transform=tsfm,
                             train=False)

BATCH_SIZE = 1

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4)

testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4)

model = LSTM_TAGGER(512, 64, 10)

inputs = next(iter(trainLoader))
# inputs = torch.randn(1, 3, 224, 224);
# model = models.resnet18()
# TODO: Import model from classifier.py and make_dot

y = model.eval()(Variable(inputs['feature']).view(1,15,-1).detach())
print(y)

g = make_dot(y, params=dict(model.named_parameters()))
g.format = 'dot'
g.render('classifier')
