"""Trim leading and trailing silence"""

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


sampling_rate = 44100

train = pd.read_csv("./data/train_curated.csv")
samp_subm = pd.read_csv("./data/sample_submission.csv")


print('Train...')
os.makedirs('./data/audio_train_trim', exist_ok=True)
for filename in tqdm(train.fname.values):
    x, sr = librosa.load('./data/train_curated/' + filename, sampling_rate)
    x = librosa.effects.trim(x)[0]
    np.save('./data/audio_train_trim/' + filename + '.npy', x)

print('Test...')
os.makedirs('./data/audio_test_trim', exist_ok=True)
for filename in tqdm(samp_subm.fname.values):
    x, sr = librosa.load('./data/test/' + filename, sampling_rate)
    x = librosa.effects.trim(x)[0]
    np.save('./data/audio_test_trim/' + filename + '.npy', x)
