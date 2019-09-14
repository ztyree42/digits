import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from dataLoader import spokenDigitDataset, ToTensor, Latent
from transforms.mixer import Mixer
from transforms.stft import ToSTFT
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

tsfm = tv.transforms.Compose([
    Mixer(),
    ToSTFT(),
    ToTensor(8)
])

trainSet = spokenDigitDataset('/home/ubuntu/projects/spokenDigits',
                              'recordings',
                              transform=tsfm,
                              train=True,
                              mixing=True)

BATCH_SIZE = 1

trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4)
                         
real_sum = 0
real_2_mom = 0
imag_sum = 0
imag_2_mom = 0

for s in trainLoader:
    real_sum += s['feature'][:,:,0,:,:].mean()
    real_2_mom += ((s['feature'][:,:, 0, :, :])**2).mean()
    imag_sum += (s['feature'][:,:, 1, :, :]).mean()
    imag_2_mom += ((s['feature'][:,:, 1, :, :])**2).mean()

real_sum /= trainLoader.__len__()
real_2_mom /= trainLoader.__len__()
imag_sum /= trainLoader.__len__()
imag_2_mom /= trainLoader.__len__()

real_std = (real_2_mom - real_sum**2)**.5
imag_std = (imag_2_mom - imag_sum**2)**.5

print(real_sum, real_std)

print(imag_sum, imag_std)