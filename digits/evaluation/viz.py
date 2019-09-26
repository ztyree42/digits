import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np

from digits.transforms.mixer import Mixer
from digits.transforms.stft import ToSTFT
from digits.transforms.normalize import Normalize
from digits.dataLoader import spokenDigitDataset, ToTensor, Latent
from torch.utils.tensorboard import SummaryWriter
from digits.features.featureSelector_stft import AE
from digits.fullDecomposer import LSTM_DECOMPOSER
import yaml

with open('/home/ubuntu/projects/digits/digits/vizParams.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

args = args['viz']
d_args = args['data']
f_args = args['transforms']
t_args = args['training']

DATA_PATH = d_args['path']
DATA_TYPE = d_args['type']
MIXING = d_args['mixing']
FULL = d_args['full']

# EMBEDDING_PATH = f_args['embedding_path']
# STEP_SIZE = f_args['step_size']
# EMBEDDING_HIDDEN = f_args['embedding_hidden']
# EMBEDDING_INPUT = f_args['embedding_input']

BATCH_SIZE = t_args['batch_size']
NUM_WORKERS = t_args['num_workers']
# INPUT_DIM = t_args['input_dim']
# HIDDEN_DIM = t_args['hidden_dim']
# OUTPUT_DIM = t_args['output_dim']
# LEARNING_RATE = t_args['learning_rate']
# WEIGHT_DECAY = t_args['weight_decay']
# EPOCHS = t_args['epochs']
# DROP_OUT = t_args['drop_out']
# MODEL_PATH = t_args['model_path']

# tsfm = tv.transforms.Compose([
#     Mixer(),
#     ToSTFT(),
#     ToTensor(STEP_SIZE, True),
#     Latent(EMBEDDING_PATH,
#            hidden_dims=EMBEDDING_HIDDEN, input_dim=EMBEDDING_INPUT,
#            full=True)
# ])
# tsfm = tv.transforms.Compose([
#     Mixer(),
#     ToSTFT(),
#     ToTensor(STEP_SIZE, True)
# ])


# def worker_init_fn(worker_id):
#     np.random.seed(np.random.get_state()[1][0] + worker_id)


trainSet = spokenDigitDataset(DATA_PATH,
                              DATA_TYPE,
                              train=True,
                              mixing=MIXING,
                              full=FULL)

# trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True,
#                          num_workers=NUM_WORKERS,
#                          worker_init_fn=worker_init_fn)
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS)


model = LSTM_DECOMPOSER(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

dae = AE(STEP_SIZE*EMBEDDING_INPUT, EMBEDDING_HIDDEN)

dae.load_state_dict(torch.load(EMBEDDING_PATH))

tstft = ToSTFT()

b = next(iter(trainLoader))
out = b['label']
out1 = out[0]*2400 + 21.5

out2 = dae.decoder(out1)
out3 = [out2[0], out2[1]]
out4 = [o.view((15, 2, 65, 8)).detach().numpy() for o in out3]
out5 = [np.concatenate(o, -1) for o in out4]
out6 = [o.transpose((1, 2, 0)) for o in out5]
out7 = [o[:, :, 0] + o[:, :, 1]*1j for o in out6]

for i, o in enumerate(out7):
    tstft.wav(o, f'sound{i}.wav')

# TEST
OUT = out7[0]
out6[0] == np.stack((np.real(OUT), np.imag(OUT)), axis=-1)
out5[0] == out6[0].transpose((2, 0, 1))
numSubArray = out5[0].shape[2] // 8
featureList = np.split(out5[0][:, :, :numSubArray*8],
                       numSubArray, axis=2)
feature = np.array(featureList)
out4[0] == feature
OUT = torch.from_numpy(out4[0])
out3[0] == OUT.view(OUT.size(0), -1)
out2[0] == dae.encoder(out3[0])

#
b = next(iter(trainLoader))
out = b['feature']
out = out[0]

b = next(iter(trainLoader))
out = b['label'][0]
out = out[0]

out1 = np.concatenate(out.detach().numpy(), -1).transpose(1, 2, 0)
out1 = out1[:, :, 0] + out1[:, :, 1]*1j
tstft.wav(out1, 'sound_mix.wav')
wav_to_spectrogram('sound_mix.wav', 'spec0.png')

out = out.view(out.size(0), -1)
out = dae.encoder(out)
out = dae.decoder(out)
out = out.view((out.size(0), 2, 65, 8))
out = np.concatenate(out.detach().numpy(), -1).transpose(1, 2, 0)
out = out[:, :, 0] + out[:, :, 1]*1j
tstft.wav(out, 'sound_mix.wav')
wav_to_spectrogram('sound_mix.wav', 'spec0.png')
