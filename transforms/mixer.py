import scipy.io.wavfile as wav
import numpy as np


# path_0 = '/home/ubuntu/projects/spokenDigits/recordings/0_jackson_0.wav'
# path_1 = '/home/ubuntu/projects/spokenDigits/recordings/1_jackson_0.wav'

# sound_0 = AudioSegment.from_file(path_0)
# sound_1 = AudioSegment.from_file(path_1)

# combined = sound_0.overlay(sound_1)

# combined.export('sound.wav', format='wav')

class Mixer():
    """Mixes two wav files with the same fs."""

    def __init__(self, rate=8000):
        self.rate = rate
    
    def __call__(self, sample):
        p0 = sample['feature'][0]
        p1 = sample['feature'][1]
        _, sound_0 = wav.read(p0)
        sound_0 = sound_0[:4000]
        sound_0 = np.pad(sound_0, 
            (0, 4000 - sound_0.shape[0]), 'constant')
        _, sound_1 = wav.read(p1)
        sound_1 = sound_1[:4000]

        sound_1 = np.pad(sound_1, 
            (0, 4000 - sound_1.shape[0]), 'constant')
        combined = .5*sound_0 + .5*sound_1
        combined = combined.astype('i2')
        return {'feature': combined, 'label': sample['label']}
