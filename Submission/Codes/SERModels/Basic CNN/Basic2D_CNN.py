# This is code for cz4042 final project
# Speech emotion recognition

# Keras
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD
# Other
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from matplotlib.pyplot import specgram
import pandas as pd
import os
import sys
import warnings
import pickle
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# load dataset
win_ts = 128
hop_ts = 64
only_radvess = True

melspec = np.load('./processed_data/mel_spec_data_1.npy')
ref = pd.read_pickle("./metadata/data_df_1.pkl")

# Split between train and test data, ratio of 1:3
X_train, X_test, y_train, y_test = train_test_split(
    melspec, ref.Emotion, test_size=0.25, shuffle=True, random_state=42)

X_train = X_train.squeeze()
X_test = X_test.squeeze()
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))
print(X_train.shape)

# Reshape X train and test, for input into model, reshape the data being flattend in feature extraction part in order to easier save
X_train = X_train.reshape(3240, 2, 1, 60, 216)
X_test = X_test.reshape(1080, 2, 1, 60, 216)

input_y = Input(shape=X_train.shape[1:], name='Input_MELSPECT')

# for test only
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Model Disign

## First LFLB (local feature learning block)
y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1')(input_y)
y = TimeDistributed(BatchNormalization(), name='BatchNorm_1')(y)
y = TimeDistributed(Activation('elu'), name='Activ_1')(y)
y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1')(y)
y = TimeDistributed(Dropout(0.3), name='Drop_1')(y)     

## Second LFLB (local feature learning block)
y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2')(y)
y = TimeDistributed(BatchNormalization(), name='BatchNorm_2')(y)
y = TimeDistributed(Activation('elu'), name='Activ_2')(y)
y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2')(y)
y = TimeDistributed(Dropout(0.3), name='Drop_2')(y)


## Third LFLB (local feature learning block)
y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3')(y)
y = TimeDistributed(BatchNormalization(), name='BatchNorm_3')(y)
y = TimeDistributed(Activation('elu'), name='Activ_3')(y)
y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3')(y)
y = TimeDistributed(Dropout(0.4), name='Drop_3')(y)

## Fourth LFLB (local feature learning block)
y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4')(y)
y = TimeDistributed(BatchNormalization(), name='BatchNorm_4')(y)
y = TimeDistributed(Activation('elu'), name='Activ_4')(y)
y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4')(y)
y = TimeDistributed(Dropout(0.4), name='Drop_4')(y)  

## Flat
y = TimeDistributed(Flatten(), name='Flat')(y)                      

## Dense layer                            

y = Flatten()(y)
y = Dense(64)(y)
y = Dropout(rate=0.5)(y)
y = BatchNormalization()(y)
y = Activation("relu")(y)
y = Dropout(rate=0.5)(y)

y = Dense(y_train.shape[1], activation='softmax', name='FC')(y)

model = Model(inputs=input_y, outputs=y)

model.compile(optimizer=Adam(lr=0.005, decay=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=1, mode='max')

# Fit model
# history = model.fit(X_train, y_train, batch_size=64, epochs=1, validation_data=(X_test, y_test), callbacks=[early_stopping])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                              batch_size=64, verbose=2, epochs=1500)

