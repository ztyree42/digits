import torch

class Normalize():
    """Normalize real and imaginary part of sequence"""
    def __init__(self, meanM, stdM, meanS, stdS, full=False):
        self.meanM = meanM
        self.stdM = stdM
        self.full = full
        self.meanS = meanS
        self.stdS = stdS
    def __call__(self, sample):
        if self.full:
            feature = sample['feature']
            feature = (feature - self.meanM) / self.stdM
            label = sample['label']
            label = (label - self.stdS) / self.stdS
            return {'feature': feature, 'label': label}
        else:
            return sample
