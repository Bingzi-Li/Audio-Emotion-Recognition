from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.models import Model
from sklearn import preprocessing
import os

dropout = 0.2
encoding_dim = 100
optimizer = "adam"
cell = "dense"
epoch = 1
feature = "logspec"

X = np.load('./processed_data/mel_spec_data_1.npy')
x_train, x_test = train_test_split(X, test_size=0.3, random_state=42)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# input normalisation
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)

# model architecture for encoding dimension = 100
if encoding_dim == 100:
	input_img = keras.Input(shape=(60*216,))
	encoded = layers.Dense(3000, activation='relu')(input_img)
	if dropout: # add dropout if the dropout set is not 0
		encoded = layers.Dropout(rate = dropout)(encoded)
	encoded = layers.Dense(100, activation='relu')(encoded)
	decoded = layers.Dense(3000, activation='relu')(encoded)
	if dropout:
		decoded = layers.Dropout(rate = dropout)(decoded)
	decoded = layers.Dense(60*216, activation='sigmoid')(decoded)

# model architecture for encoding dimension = 300
elif encoding_dim == 300:
	input_img = keras.Input(shape=(60*216,))
	encoded = layers.Dense(3000, activation='relu')(input_img)
	if dropout:
		encoded = layers.Dropout(rate = dropout)(encoded)
	encoded = layers.Dense(300, activation='relu')(encoded)
	decoded = layers.Dense(3000, activation='relu')(encoded)
	if dropout:
		decoded = layers.Dropout(rate = dropout)(decoded)
	decoded = layers.Dense(60*216, activation='sigmoid')(decoded)

# build autoencoder
autoencoder = keras.Model(input_img, decoded)
print(autoencoder.summary())

# build encoder
encoder = keras.Model(input_img, encoded)

# build decoder
encoded_input = keras.Input(shape=(encoding_dim,))

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# train autoencoder
history = autoencoder.fit(x_train, x_train, # the same data is used as input and target output for training an autoencoder
                epochs=epoch,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, x_test))

#save results
if not os.path.exists('./autoencoder_loss/'):
    os.mkdir('./autoencoder_loss/')
    
if dropout:
	encoder.save('./autoencoder_loss/3_layer_encoder-{}-dropout={}-{}-{}-{}.hdf5'.format(cell,dropout,optimizer,feature,encoding_dim))
else:
	encoder.save('./autoencoder_loss/3_layer_encoder-no_dropout={}-{}-{}-{}.hdf5'.format(cell,optimizer,feature,encoding_dim))
import pickle
res = {"loss":history.history["loss"],"val_loss" : history.history["val_loss"]}
if dropout:
	pickle.dump(res,open("./autoencoder_loss/3_layer_encoder-{}-dropout={}-{}-{}-{}.pkl".format(cell,dropout,optimizer,feature,encoding_dim),"wb"))
else:
	pickle.dump(res,open("./autoencoder_loss/3_layer_encoder-no_dropout-abs{}-{}-{}-{}.pkl".format(cell,optimizer,feature,encoding_dim),"wb"))

