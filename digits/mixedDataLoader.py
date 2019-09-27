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
from digits.features.featureSelector_stft import AE
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from digits.transforms.mixer import Mixer
from digits.transforms.stft import ToSTFT
import scipy.io.wavfile as wav
import scipy.signal as signal


class MixedDigits(Dataset):

    def __init__(self, digitDir, digitType='mixedSpectrograms',
                 transform=None, train=True, rate=8000):
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
        self.rate = rate
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
        pathList = digitList
        digitList = [x.replace('.wav.', '').replace(
            '.png', '') for x in digitList]
        digitList = [x.split('-') for x in digitList]
        digitList = [[el for x in y for el in (
            x[0], x.split('_')[-1])] for y in digitList]
        # digitList = [(x[0], x) for x in digitList]
        df = pd.DataFrame.from_records(digitList, columns=[
            'label1', 'instance1', 'label2', 'instance2'
        ])
        df['path'] = pathList
        df['label1'] = df['label1'].astype(int)
        df['label2'] = df['label2'].astype(int)
        df['instance1'] = df['instance1'].astype(int)
        df['instance2'] = df['instance2'].astype(int)
        if self.train:
            df = df[((df['instance1'] > 2) | (df['instance2'] > 2))]
        else:
            df = df[((df['instance1'] <=2) & (df['instance2'] <=2))]
        return df

    def __len__(self):
            return len(self.digitFrame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = join(self.digitDir, self.digitType, self.digitFrame.iloc[
            idx, -1
        ])
        if self.digitType == 'mixedRecordings':
            sample = self._getRecording(path, idx)
        if self.digitType == 'mixedSpectrograms':
            sample = self._getSpectrogram(path, idx)
        if self.transform is not None:
            sample['feature'] = self.transform(sample['feature'])
        return sample

    def _getRecording(self, path, idx):
        _, samples = wav.read(path)
        label = self.digitFrame.iloc[idx, [0,2]]

        sample = {'feature': samples, 'label': label.tolist()}
        return sample

    def _getSpectrogram(self, path, idx):
        image = np.delete(io.imread(path), [1, 2, 3], 2)

        label = self.digitFrame.iloc[idx, [0,2]]

        sample = {'feature': image, 'label': label.tolist()}

        return sample

def showBatch(batch):
    """Show spectrograms from batch."""
    feature_batch, label_batch = \
        batch['feature'], batch['label']

    batch_size = len(feature_batch)
    feature_batch = feature_batch.view(batch_size, 1, 64, 64)
    feature_batch = feature_batch.transpose(2, 3)
    feature_size = feature_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(feature_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.text(i*feature_size + (i+1) * grid_border_size + 30, 10 + grid_border_size,
                 label_batch[i].item(), horizontalalignment='center',
                 verticalalignment='center', color='blue', size='xx-large')
        plt.title("Batch from dataLoader")
