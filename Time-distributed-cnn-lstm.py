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
from matplotlib.pyplot import specgram
import pandas as pd
import os
import sys
import warnings
import pickle
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

class attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)

from tensorflow.keras.optimizers import SGD, Adam

# Split spectrogram into frames
def frame(x, win_step=128, win_size=64):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames


if __name__ == '__main__':
    # load dataset
    win_ts = 128
    hop_ts = 64
    only_radvess = True
    data_feature = 'mfcc'
    
    extra_name = 'ravdess-' if only_radvess else ''
    mfcc = np.load(f'./processed_data/{extra_name}{data_feature}.npy')
    ref = pd.read_csv("./metadata/Data_path.csv")
    if only_radvess:
        ref = ref.loc[ref['source'] == 'RAVDESS']

    # Split between train and test
    X_train, X_test, y_train, y_test = train_test_split(
        mfcc, ref.labels, test_size=0.25, shuffle=True, random_state=42)

    X_train = X_train.squeeze()
    X_test = X_test.squeeze()
    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(y_train))
    y_test = to_categorical(lb.fit_transform(y_test))
    print(X_train.shape)
#     Frame for TimeDistributed model
    X_train = frame(X_train, hop_ts, win_ts)
    X_test = frame(X_test, hop_ts, win_ts)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , X_train.shape[2], X_train.shape[3], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , X_test.shape[2], X_test.shape[3], 1)
    

    input_y = Input(shape=X_train.shape[1:], name='Input_MELSPECT')

    ## First LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT')(input_y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)     

    ## Second LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT')(y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)


    ## Third LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT')(y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

    ## Fourth LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT')(y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)  

    ## Flat
    y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)                      
                                   
    # Apply 2 LSTM layer and one FC
    y = LSTM(256, return_sequences=True, dropout=0.2, name='LSTM_1')(y)
    y = attention(return_sequences=False)(y)


    y = Flatten()(y)
    y = Dense(64)(y)
    y = Dropout(rate=0.2)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(rate=0.2)(y)
    
    y = Dense(y_train.shape[1], activation='softmax', name='FC')(y)

    model = Model(inputs=input_y, outputs=y)

    model.compile(optimizer=Adam(lr=0.001, decay=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=1, mode='max')

    # Fit model
    # history = model.fit(X_train, y_train, batch_size=64, epochs=1, validation_data=(X_test, y_test), callbacks=[early_stopping])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                  batch_size=64, verbose=2, epochs=1000)