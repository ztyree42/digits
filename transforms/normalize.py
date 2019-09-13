import torch

class Normalize():
    """Normalize real and imaginary part of sequence"""
    def __init__(self):
        pass
    def __call__(self, sample):
        data = sample['feature']
        data[:,:,0,:] = (data[:,:,0,:] - .16) / 69
        data[:,:,1,:] = (data[:,:,1,:] - 0) / 68
        return {'feature': data, 'label': sample['label']}
