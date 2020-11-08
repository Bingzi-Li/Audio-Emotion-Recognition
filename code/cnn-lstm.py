# Keras
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import (Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                                     MaxPooling2D, Reshape, Activation, Input, LSTM, TimeDistributed)

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other
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
data_feature = 'mfcc'
only_radvess = False


def get_2d_conv_model(n):
    ''' Create a standard deep 2D convolutional neural network'''
    nclass = 12

    # ---------- This is another modelling style -----------#
    # inp = Input(shape=(n, 216, 1))
    # print('Each batch: ', inp.shape)
    # inp = Reshape(target_shape=(216, 30, 1, 1), input_shape=(30, 216, 1))(inp)
    # x = TimeDistributed(Conv2D(32, (4, 10), padding="same",
    #                            input_shape=(30, 1, 1)))(inp)
    # x = TimeDistributed(MaxPooling2D())(x)
    # x = TimeDistributed(Flatten())(x)

    # x = LSTM(20)(x)
    # x = Dropout(rate=0.2)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(rate=0.2)(x)
    # out = Dense(nclass, activation='softmax')(x)
    # model = models.Model(inputs=inp, outputs=out)

    # define cnn model
    cnn = Sequential()
    cnn.add(Input(shape=(30, 216, 1)))
    cnn.add(Reshape(target_shape=(216, 30, 1, 1), input_shape=(30, 216, 1)))
    cnn.add(Conv2D(32, (4, 10), padding="same",
                   input_shape=(30, 1, 1)))
    cnn.add(MaxPooling2D(pool_size=(1, 2)))
    cnn.add(Flatten())

    # define LSTM model
    model = Sequential()
    model.add(TimeDistributed(cnn))
    model.add(LSTM(20))
    model.add(Dropout(rate=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(nclass, activation='softmax'))

    opt = optimizers.Adam(0.001)
    model.compile(
        optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


if __name__ == '__main__':
    # load dataset

    extra_name = 'ravdess-' if only_radvess else ''
    data = np.load(f'./processed_data/{extra_name}{data_feature}.npy')
    ref = pd.read_csv("./metadata/Data_path.csv")
    if only_radvess:
        ref = ref.loc[ref['source'] == 'RAVDESS']

    # Split between train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data, ref.labels, test_size=0.25, shuffle=True, random_state=42)

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
    model.save(f'./models/2DCNNlstm-{extra_name}{data_feature}')
    if not os.path.exists('./histories'):
        os.mkdir('./histories')
    hist_path = f'./histories/2DCNNlstm-{extra_name}{data_feature}'
    with open(hist_path, 'wb') as file_pi:
        pickle.dump(model_history.history, file_pi)
