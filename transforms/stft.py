import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

# rate, samples = wav.read('/home/ubuntu/projects/spokenDigits/recordings/1_jackson_0.wav')

# OVERLAP = 96
# PERSEG = 128

# signal.check_NOLA('hann', PERSEG, OVERLAP)

# f, t, Zxx = signal.stft(samples, fs=rate, nperseg=PERSEG, noverlap=OVERLAP)

# times, arr = signal.istft(Zxx, fs=rate, nperseg=PERSEG, noverlap=OVERLAP)

# wav.write('/home/ubuntu/Desktop/test.wav', rate=rate,
#             data=arr.astype(np.dtype('i2')))

# # plt.pcolormesh(t, f, np.abs(Zxx))
# plt.subplot(211)
# plt.plot(times, arr)
# plt.subplot(212)
# plt.specgram(arr, Fs=rate)
# plt.show()

# plt.subplot(211)
# plt.plot(samples)
# plt.subplot(212)
# plt.specgram(samples, Fs=rate)
# plt.show()


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
        Zxx = np.stack((np.real(Zxx), np.imag(Zxx)), axis=-1)
        return {'feature': Zxx, 'label': sample['label']}
