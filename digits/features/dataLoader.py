from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from os import listdir
from os.path import join, isfile
import re
from skimage import io, transform
import pandas as pd
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)


class spokenDigitDataset(Dataset):

    def __init__(self, digitDir, digitType='spectrograms', 
        transform=None, train=True, sanity=False, full=False):
        """
        Args:
            digitDir (string): base dir of spoken digits
            type (string): either 'recordings' or 'spectrogram'
            transform (callable, optional): transform to be applied
                on a sample.
        """
        self.digitDir = digitDir
        self.digitType = digitType
        self.transform = transform
        self.train = train
        self.sanity = sanity
        self.full = full
        self.digitFrame = self._frameBuilder()

    def instanceF(self, path):
        path = re.sub('.png', '', path)
        path = re.sub('.wav', '', path)
        path = int(re.search('([^_]*)$', path).group(0))
        return path

    def _frameBuilder(self):
        digitList = listdir(join(self.digitDir, self.digitType))
        digitList = [f for f in digitList if isfile(join(
            self.digitDir,
            self.digitType,
            f
        ))]
        digitList = [(x[0], x) for x in digitList]
        df = pd.DataFrame.from_records(digitList, columns=[
            'label', 'path'
        ])
        df['label'] = df['label'].astype(int)
        df['instance'] = df.apply(
            lambda row: self.instanceF(row.path), axis=1)
        if self.full:
            pass
        elif self.sanity:
            df = df.head(12)
        elif self.train:
            df = df[(df['instance'] > 4)]
        else:
            df = df[(df['instance'] <= 4)]
        return df

    def __len__(self):
        return len(self.digitFrame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = join(self.digitDir, self.digitType, self.digitFrame.iloc[
            idx, 1
        ])
        if self.digitType == 'recordings':
            return self._getRecording(path, idx)
        if self.digitType == 'spectrograms':
            return self._getSpectrogram(path, idx)

    def _getRecording(self, path, idx):
        _, samples = wav.read(path)
        return samples

    def _getSpectrogram(self, path, idx):
        image = np.delete(io.imread(path), [1, 2, 3], 2)

        if self.transform is not None:
            image = self.transform(image)

        return image

# todo: this class is only appropriate to call on spectrograms atm
#       likely needs to be made more general (factory?)


class ToTensor(object):
    """Converts ndarrays in sample to Tensors."""

    def __init__(self, stepSize):
        self.stepSize = stepSize

    def __call__(self, sample):
        idx = np.random.randint(0, sample.shape[1] - self.stepSize)
        
        sample = sample[:,idx:(idx + self.stepSize), :]
        # swap color axis from np (HxWxC) to torch (CxHxW)
        sample = sample.transpose((2, 0, 1))
        sample = (sample - 127.5)/127.5
        return torch.from_numpy(sample).squeeze().type(torch.FloatTensor)