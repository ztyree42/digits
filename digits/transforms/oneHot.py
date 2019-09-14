import torch
import torch.nn.functional as F
import numpy as np

class OneHot():
    """ Converts labels to one-hot encodings. """
    def __init__(self, num_classes):
        self.nclass = num_classes
    
    def multiToOne(self, t):
        def zoro(t):
            return min(t,torch.from_numpy(np.array(1)))
        
        t = list(map(zoro, t.unbind(0)))
        return torch.stack(t, 0)

    def __call__(self, sample):



        labels = sample['label']
        labels = F.one_hot(labels, self.nclass)
        labels = labels.sum(dim=0)
        labels = self.multiToOne(labels)
        labels = labels.float()
        s = {'feature': sample['feature'],
             'label': labels}
        return s
