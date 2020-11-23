# This code is for doing feature extraction for cz4042 final project
# Speech emotion recognition

import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
import matplotlib.pyplot as plt

# Labels for emotions
EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'}
# Data path to load input audio dataset
DATA_PATH = './inputs/Audio_Speech_Actors_01-24/'
SAMPLE_RATE = 44100

# Create a new dataframe and load all audio input files inside, with four columns,'Emotion', 'Intensity', 'Gender','Path'

data = pd.DataFrame(columns=['Emotion', 'Intensity', 'Gender','Path'])

for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        identifiers = filename.split('.')[0].split('-')
        emotion = (int(identifiers[2]))
        if emotion == 8:
            emotion = 0
        if int(identifiers[3]) == 1:
            emotion_intensity = 'normal' 
        else:
            emotion_intensity = 'strong'
        if int(identifiers[6])%2 == 0:
            gender = 'female'
        else:
            gender = 'male'
        
        data = data.append({"Emotion": emotion,
                            "Intensity": emotion_intensity,
                            "Gender": gender,
                            "Path": file_path
                             },
                             ignore_index = True
                          )
print("number of files is {}".format(len(data)))

# Method to calculate Mel-spectrogram using librosa library
def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length = 512,
                                              window='hamming',
                                              hop_length = 258,
                                              n_mels=60,
                                              fmax=sample_rate/2
                                             )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Test function for feature extraction
audio, sample_rate = librosa.load(data.loc[0,'Path'], duration=3, offset=0.5,sr=SAMPLE_RATE)
signal = np.zeros((int(SAMPLE_RATE*3,)))
signal[:len(audio)] = audio
mel_spectrogram = getMELspectrogram(signal, SAMPLE_RATE)
librosa.display.specshow(mel_spectrogram, y_axis='mel', x_axis='time')
print('Mel-spectrogram shape: ',mel_spectrogram.shape)

# Generate mel-spectrogram data for all raw inputs files 
mel_spectrograms = []
signals = []
for i, file_path in enumerate(data.Path):
    audio, sample_rate = librosa.load(file_path, duration=2.5, offset=0.5, sr=SAMPLE_RATE)
    signal = np.zeros((int(SAMPLE_RATE*3,)))
    signal[:len(audio)] = audio
    signals.append(signal)
    mel_spectrogram = getMELspectrogram(signal, sample_rate=SAMPLE_RATE)
    mel_spectrograms.append(mel_spectrogram)
    print("\r Processed {}/{} files".format(i,len(data)),end='')

print('Mel-spectrogram shape: ',mel_spectrogram.shape)

# Function used for data augmentation
def addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30): 
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0**(num_bits-1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise 
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K  
    # Generate noisy signal
    return signal + K.T * noise

    # Data augmentation
for i,signal in enumerate(signals):
    augmented_signals = addAWGN(signal)
    for j in range(augmented_signals.shape[0]):
        mel_spectrogram = getMELspectrogram(augmented_signals[j,:], sample_rate=SAMPLE_RATE)
        mel_spectrograms.append(mel_spectrogram)
        data = data.append(data.iloc[i], ignore_index=True)
    print("\r Processed {}/{} files".format(i,len(signals)),end='')

print('Mel-spectrogram shape: ',mel_spectrogram.shape)

# Save data pickle file for further use
if not os.path.exists('./metadata/'):
    os.mkdir('./metadata/')
data.to_pickle("./metadata/data_df_1.pkl")

# Method to split data into chunks
def splitIntoChunks(mel_spec,win_size,stride):
    t = mel_spec.shape[1]
    num_of_chunks = int(t/stride)
    chunks = []
    for i in range(num_of_chunks):
        chunk = mel_spec[:,i*stride:i*stride+win_size]
        if chunk.shape[1] == win_size:
            chunks.append(chunk)
    return np.stack(chunks,axis=0)

# Get chunks of Mel-spectrogram data
mel_spectrograms_chunked = []
for mel_spec in mel_spectrograms:
    chunks = splitIntoChunks(mel_spec, win_size=216,stride=160)
    mel_spectrograms_chunked.append(chunks)
print("Number of chunks is {}".format(chunks.shape[0]))


mel_spectrograms_chunked = np.array(mel_spectrograms_chunked)
print('Mel-spectrogram-chunked shape: ',mel_spectrograms_chunked.shape)
arr = np.array(mel_spectrograms_chunked)
reshape = arr.reshape(4320, 2*60*216)
if not os.path.exists('./processed_data/'):
    os.mkdir('./processed_data/')
np.save("./processed_data/mel_spec_data_1.npy", reshape)