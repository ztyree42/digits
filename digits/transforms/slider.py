import torch

class Slider():
    def __init__(self, size, step):
        self.size = size
        self.step = step

    def __call__(self, inp):
        inp = inp.squeeze().unfold(0, self.size, self.step)
        return inp
