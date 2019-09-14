import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

class ToSTFT():
    """ Converts wav to stft and back. """

    def __init__(self, noverlap=96, nperseg=128, rate=8000):
        self.noverlap = noverlap
        self.nperseg = nperseg
        self.rate = rate

    def stft(self, samples):
        f, t, Zxx = signal.stft(
            samples, fs=self.rate, nperseg=self.nperseg,
            noverlap=self.noverlap)
        return f, t, Zxx

    def istft(self, Zxx):
        times, arr = signal.istft(
            Zxx, fs=self.rate, nperseg=self.nperseg,
            noverlap=self.noverlap)
        return times, arr.astype(np.dtype('i2'))

    def wav(self, Zxx, outPath):
        _, arr = self.istft(Zxx)
        wav.write(outPath, rate=self.rate, data=arr)

    def __call__(self, sample):
        samples = sample['feature']
        Zxx = self.stft(samples)[2]
        Zxx = np.stack((np.abs(Zxx), np.angle(Zxx)), axis=-1)
        return {'feature': Zxx, 'label': sample['label']}
