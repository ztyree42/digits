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

class spokenDigitDataset(Dataset):
    
    def __init__(self, digitDir, digitType = 'spectrograms', 
                 transform = None, train = True, sanity=False, 
                 rate=8000, mixing=False, full=False):
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
        self.rate = rate
        self.mix = mixing
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
                lambda row: self.instanceF(row.path), axis =1)
        if self.sanity:
            df = df.head(12)
        elif self.train:
            df = df[(df['instance'] > 4)]
        else:
            df = df[(df['instance'] <= 4)]
        return df
    
    def __len__(self):
        if self.mix:
            return 2*len(self.digitFrame)
        else:
            return len(self.digitFrame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.mix:
            sample = self.__getMix(idx)
        else:
            path = join(self.digitDir, self.digitType, self.digitFrame.iloc[
                idx, 1
            ])
            if self.digitType == 'recordings':
                sample = self._getRecording(path, idx)
            if self.digitType == 'spectrograms':
                sample = self._getSpectrogram(path, idx)
        if self.transform is not None:
            sample['feature'] = self.transform(sample['feature'])
        return sample
    
    def __getMix(self, idx):
        length = (self.__len__() // 2)
        idxs = [idx % length, np.random.randint(0, length)]
        for i in idxs:
            if torch.is_tensor(i):
                i = i.tolist()
        if idx >= length:
            idxs.reverse()
        idx0 = idxs[0]
        idx1 = idxs[1]
        p0 = join(self.digitDir, self.digitType, self.digitFrame.iloc[
            idx0, 1
        ])
        p1 = join(self.digitDir, self.digitType, self.digitFrame.iloc[
            idx1, 1
        ])
        l0 = self.digitFrame.iloc[idx0, 0]
        l1 = self.digitFrame.iloc[idx1, 0]
        label = [l0, l1]
        if self.full:
            label = [p0, p1]
            for i,s in enumerate(label):
                _, s = wav.read(s)
                s = s[:4000]
                s = np.pad(s, (0, 4000 - s.shape[0]), 'constant')
                _, _, s = signal.stft(s, 
                    fs=8000, nperseg=128, noverlap=96) 
                # s = np.stack((np.real(s), np.imag(s)), axis=-1)
                s = (np.real(s)**2 + np.imag(s)**2)**.5
                label[i] = s
        sample = {'feature': [p0, p1], 'label': label}
        return sample

    def _getRecording(self, path, idx):
        _, samples = wav.read(path)
        samples = np.pad(samples, 
            (0, self.rate - samples.shape[0]), 'constant')
        label = self.digitFrame.iloc[idx, 0]

        sample = {'feature': samples, 'label': label}
        return sample
    
    def _getSpectrogram(self, path, idx):
        image = np.delete(io.imread(path), [1,2,3], 2)

        label = self.digitFrame.iloc[idx, 0]

        sample = {'feature': image, 'label': label}
        
        return sample

class ToTensor(object):
    """Converts ndarrays in sample to Tensors."""

    def __init__(self, stepSize, full=False):
        self.stepSize = stepSize
        self.full = full

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']
        # swap color axis from np (HxWxC) to torch (CxHxW)
        # feature = feature.transpose((1,0))
        numSubArray = feature.shape[1] // self.stepSize
        featureList = np.split(feature[:,:numSubArray*self.stepSize], 
            numSubArray, axis=1)
        # featureList = [x.reshape((1,feature.shape[1])) for x in featureList]
        feature = np.array(featureList)
        if not self.full:
            return {'feature': torch.from_numpy(
                        feature).squeeze().type(torch.FloatTensor),
                    'label': torch.from_numpy(np.array(label))}
        else:
            for i, l in enumerate(label):
                # l = l.transpose((1,0))
                numSubArray = l.shape[1] // self.stepSize
                lList = np.split(l[:, :numSubArray*self.stepSize], 
                    numSubArray, axis=1)
                l = np.array(lList)
                l = torch.from_numpy(l).squeeze().type(torch.FloatTensor)
                label[i] = l
            return {'feature': torch.from_numpy(
                feature).squeeze().type(torch.FloatTensor),
                'label': label}

class Latent():
    """Converts images to latent space via simple autoencoder."""
    def __init__(self, path, step_size=8, 
                 hidden_dims = [128, 64, 32, 16], 
                 input_dim=64, full=False):
        self.full = full
        model = AE(input_dim*step_size, hidden_dims)
        model.load_state_dict(torch.load(path))
        # model.load_state_dict(torch.load(path, 
        #     map_location=lambda storage, loc: storage))
        model.eval()
        self.encoder = model.encoder 
        
    def __call__(self, sample):
        x = sample['feature'].view(sample['feature'].size(0), -1)
        if not self.full:
            return {'feature': self.encoder(x).detach(),
                    'label': sample['label']}
        else:
            label = sample['label']
            for i, l in enumerate(label):
                l = l.view(l.size(0), -1)
                label[i] = l
            label = [self.encoder(l).detach() for l in label]
            label = torch.stack(label)
            return {'feature': self.encoder(x).detach(),
                    'label': label}

def showBatch(batch):
    """Show spectrograms from batch."""
    feature_batch, label_batch = \
        batch['feature'], batch['label']
    
    batch_size = len(feature_batch)
    feature_batch = feature_batch.view(4,1,64,64)
    feature_batch = feature_batch.transpose(2,3)
    feature_size = feature_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(feature_batch)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    
    for i in range(batch_size):
        plt.text(i*feature_size + (i+1) * grid_border_size + 30, 10 + grid_border_size, 
                 label_batch[i].item() , horizontalalignment='center',
             verticalalignment='center', color='blue', size='xx-large')
        plt.title("Batch from dataLoader")
