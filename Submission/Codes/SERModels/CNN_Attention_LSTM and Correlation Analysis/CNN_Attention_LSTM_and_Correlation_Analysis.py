# This is code for cz4042 final project
# Speech emotion recognition

# Keras
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
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

# Define attention layer
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

# Load dataset
win_ts = 128
hop_ts = 64
melspec = np.load(path + 'mel_spec_data_1.npy')
ref = pd.read_pickle(path + "data_df_1.pkl")
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
                                
# Apply 2 LSTM layer and one FC
y = LSTM(256, return_sequences=True, dropout=0.2, name='LSTM_1')(y)
y = attention(return_sequences=False)(y)


y = Flatten()(y)
y = Dense(64)(y)
y = Dropout(rate=0.4)(y)
y = BatchNormalization()(y)
y = Activation("relu")(y)
y = Dropout(rate=0.4)(y)

y = Dense(y_train.shape[1], activation='softmax', name='FC')(y)

model = Model(inputs=input_y, outputs=y)

model.compile(optimizer=Adam(lr=0.005, decay=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model
# history = model.fit(X_train, y_train, batch_size=64, epochs=1, validation_data=(X_test, y_test), callbacks=[early_stopping])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                              batch_size=64, verbose=2, epochs=1500)

# plot curves - cnn lstm attention training and test accuracy
fig=plt.gcf()

history.history['accuracy'] = pd.Series(history.history['accuracy']).rolling(window=10).mean()
history.history['val_accuracy'] = pd.Series(history.history['val_accuracy']).rolling(window=10).mean()

axes = plt.gca()
axes.set_ylim([0.35,1])

plt.plot(history.history['accuracy'], label='cnn lstm attention training acc', c='#1f77b4')
plt.plot(history.history['val_accuracy'], label='cnn lstm attention val acc', c='#ff7f0e')

plt.title('cnn lstm attention training and testing accuracy')
plt.ylabel('accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")

fig.set_size_inches(13, 8, forward=True)
plt.show()

# emotion labels used for generating evaluation pictures
EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'} # surprise je promenjen sa 8 na 0
# confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sn

predictions = model.predict(X_test)
cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
names = [EMOTIONS[ind] for ind in range(len(EMOTIONS))]
df_cm = pd.DataFrame(cm, index=names, columns=names)
plt.figure(figsize=(15,12))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.show()

# correlation between emotion intensity and prediction
correct_strong = 0
correct_normal = 0
wrong_strong = 0
wrong_normal = 0
for i in range(len(X_test)):
    intensity = ref.loc[i,'Intensity']
    if y_test[i].argmax() == predictions[i].argmax(): # correct prediction
        if  intensity == 'normal':
            correct_normal += 1
        else:
            correct_strong += 1
    else: # wrong prediction
        if intensity == 'normal':
            wrong_normal += 1
        else:
            wrong_strong += 1
array = np.array([[wrong_normal,wrong_strong],[correct_normal,correct_strong]])
df = pd.DataFrame(array,['wrong','correct'],['normal','strong'])
plt.figure(figsize=(12,8))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df, annot=True, annot_kws={"size": 16}) # font size
plt.show()

# correlation between gender and corectness
correct_male = 0
correct_female = 0
wrong_male = 0
wrong_female = 0
for i in range(len(X_test)):
    gender = ref.loc[i,'Gender']
    if y_test[i].argmax() == predictions[i].argmax(): # correct prediction
        if  gender == 'male':
            correct_male += 1
        else:
            correct_female += 1
    else: # wrong prediction
        if gender == 'male':
            wrong_male += 1
        else:
            wrong_female += 1
array = np.array([[wrong_male,wrong_female],[correct_male,correct_female]])
df = pd.DataFrame(array,['wrong','correct'],['male','female'])
plt.figure(figsize=(12,8))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df, annot=True, annot_kws={"size": 16}) # font size
plt.show()
