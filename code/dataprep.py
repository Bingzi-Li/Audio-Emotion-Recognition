from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob
import os
import sys
import IPython.display as ipd  # To play sound in the notebook
import warnings
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def prepare_data(df, n, feature, aug=False):
    X = np.empty(shape=(df.shape[0], n, 216, 1))
    input_length = sampling_rate * audio_duration

    cnt = 0
    for fname in tqdm(df.path):
        file_path = fname
        data, _ = librosa.load(file_path, sr=sampling_rate, res_type="kaiser_fast", duration=2.5, offset=0.5
                               )

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, int(input_length) -
                                 len(data) - offset), "constant")

        # which feature?
        if feature == 'MFCC':
            # MFCC extraction
            MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
            MFCC = np.expand_dims(MFCC, axis=-1)
            X[cnt, ] = MFCC

        elif feature == 'Log':
            # Log-melspectogram
            melspec = librosa.feature.melspectrogram(data, n_mels=n_melspec)
            logspec = librosa.amplitude_to_db(melspec)
            logspec = np.expand_dims(logspec, axis=-1)
            X[cnt, ] = logspec
        else:
            raise Exception('The feature extraction method is undefined.')

        cnt += 1

    return X


if __name__ == '__main__':
    ref = pd.read_csv("../metadata/Data_path.csv")

    # MFCC
    sampling_rate = 44100
    audio_duration = 2.5
    n_mfcc = 30
    mfcc = prepare_data(ref, n=n_mfcc, feature='MFCC')
    if not os.path.exists('../processed_data/'):
        os.mkdir('../processed_data/')
    with open("../processed_data/ravdess-mfcc.npy", "wb") as f:
        np.save(file=f, arr=mfcc)

    # Log-melspectogram
    sampling_rate = 44100
    audio_duration = 2.5
    n_melspec = 60
    mel_specgram = prepare_data(ref,  n=n_melspec, feature='Log')

    if not os.path.exists('../processed_data/'):
        os.mkdir('../processed_data/')
    with open("../processed_data/ravdess-logspec.npy", "wb") as f:
        np.save(file=f, arr=mel_specgram)
