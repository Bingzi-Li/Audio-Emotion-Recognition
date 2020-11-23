from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.models import Model
from sklearn import preprocessing
import os

dropout = 0 
encoding_dim = 100
optimizer = "adam"
cell = "GRU"
epoch = 500
feature = "logspec"

X = np.load('./processed_data/mel_spec_data_1.npy')
x_train, x_test = train_test_split(X, test_size=0.3, random_state=42)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# input normalisation
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)

# split the sequence data for rnn layer. 60 is used as the n_melspec used during feature extraction is set as 60
x_train_input = x_train.reshape(len(x_train), 60,-1)
x_test_input = x_test.reshape(len(x_test), 60,-1)

# model architecture for encoding dimension = 100
if encoding_dim == 100:
	input_img = keras.Input(shape = x_train_input.shape[1:])
    
	encoded = layers.GRU(100, activation='relu',return_sequences = True)(input_img)
	if dropout:# add dropout if the dropout set is not 0
		encoded = layers.Dropout(rate = dropout)(encoded)


	decoded = layers.GRU(100, activation='relu',return_sequences = True)(encoded)
	if dropout:
		decoded = layers.Dropout(rate = dropout)(decoded)



	decoded = layers.Flatten()(decoded)
	decoded = layers.Dense(60*216, activation='sigmoid')(decoded)

# model architecture for encoding dimension = 300
elif encoding_dim == 300:
	input_img = keras.Input(shape = x_train_input.shape[1:])
    
	encoded = layers.GRU(300, activation='relu',return_sequences = True)(input_img)
	if dropout:
		encoded = layers.Dropout(rate = dropout)(encoded)

	decoded = layers.GRU(300, activation='relu',return_sequences = True)(encoded)
	if dropout:
		decoded = layers.Dropout(rate = dropout)(decoded)

	decoded = layers.Flatten()(decoded)
	decoded = layers.Dense(60*216, activation='sigmoid')(decoded)


# build autoencoder
autoencoder = keras.Model(input_img, decoded)
print(autoencoder.summary())

# build encoder
encoder = keras.Model(input_img, encoded)

# build decoder
encoded_input = keras.Input(shape=(encoding_dim,))

# compile autoencoder
autoencoder.compile(optimizer=optimizer, loss='mse')

# train autoencoder
history = autoencoder.fit(x_train_input, x_train, # the target data is flattened in correspondence to the last flatten layer in the decoder
                epochs=epoch,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test_input, x_test))

#save results
if not os.path.exists('./autoencoder_loss/'):
    os.mkdir('./autoencoder_loss/')
    
if dropout:
	encoder.save('./autoencoder_loss/1_layer_encoder-{}-dropout={}-{}-{}-{}.hdf5'.format(cell,dropout,optimizer,feature,encoding_dim))
else:
	encoder.save('./autoencoder_loss/1_layer_encoder-no_dropout_{}-{}-{}-{}.hdf5'.format(cell,optimizer,feature,encoding_dim))
import pickle
res = {"loss":history.history["loss"],"val_loss" : history.history["val_loss"]}
if dropout:
	pickle.dump(res,open("./autoencoder_loss/1_layer_encoder-{}-dropout={}-{}-{}-{}.pkl".format(cell,dropout,optimizer,feature,encoding_dim),"wb"))
else:
	pickle.dump(res,open("./autoencoder_loss/1_layer_encoder-no_dropout-{}-{}-{}-{}.pkl".format(cell,optimizer,feature,encoding_dim),"wb"))

print(min(history.history["val_loss"]))
