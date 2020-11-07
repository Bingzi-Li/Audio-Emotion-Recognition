# Keras
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                                     GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other
from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import os
import sys
import warnings
import pickle
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# constants
no_epochs = 50
batch_size = 16
sampling_rate = 44100
audio_duration = 2.5
n_mfcc = 30
n_spec = 60
# logspec or mfcc
data_feature = 'logspec'
only_radvess = False


def Conv2DBlock(x):
    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    return x


def get_2d_conv_model(n):
    ''' Create a standard deep 2D convolutional neural network'''
    nclass = 12
    # 2D matrix of 30 MFCC bands by 216 audio length.
    inp = Input(shape=(n, 216, 1))
    print('Each batch: ', inp.shape)

    # Conv blocks
    x = inp
    for i in range(4):
        x = Conv2DBlock(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.2)(x)

    out = Dense(nclass, activation=softmax)(x)
    model = models.Model(inputs=inp, outputs=out)

    opt = optimizers.Adam(0.001)
    model.compile(
        optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


if __name__ == '__main__':
    # load dataset

    extra_name = 'ravdess-' if only_radvess else ''
    mfcc = np.load(f'./processed_data/{extra_name}{data_feature}.npy')
    ref = pd.read_csv("./metadata/Data_path.csv")
    if only_radvess:
        ref = ref.loc[ref['source'] == 'RAVDESS']

    # Split between train and test
    X_train, X_test, y_train, y_test = train_test_split(
        mfcc, ref.labels, test_size=0.25, shuffle=True, random_state=42)

    print('X_train: ', X_train.shape)

    # one hot encode the category
    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(y_train))
    y_test = to_categorical(lb.fit_transform(y_test))

    # data Normalization (standard)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std

    print('X_train: ', X_train.shape)

    # Build CNN model
    if data_feature == 'mfcc':
        model = get_2d_conv_model(n=n_mfcc)
    else:
        model = get_2d_conv_model(n=n_spec)
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
    model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                              batch_size=batch_size, verbose=2, epochs=no_epochs)

    # save model and history
    if not os.path.exists('./models'):
        os.mkdir('./models')
    model.save(f'./models/2DCNN-{extra_name}{data_feature}')
    if not os.path.exists('./histories'):
        os.mkdir('./histories')
    hist_path = f'./histories/2DCNN-{extra_name}{data_feature}'
    with open(hist_path, 'wb') as file_pi:
        pickle.dump(model_history.history, file_pi)
