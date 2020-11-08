from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.utils import to_categorical

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
import seaborn as sns
import warnings
import pickle
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# mfcc or logspec
data_feature = 'logspec'
only_radvess = False
extra_name = 'ravdess-' if only_radvess else ''


def print_confusion_matrix(confusion_matrix, class_names, figsize=(11, 8), fontsize=10):
    '''Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    '''
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.set_ylim(12, 0)
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./plots/2DCNN-{extra_name}{data_feature}-confusion.png')


class get_results:

    def __init__(self, model_history, model, X_test, y_test, labels):
        self.model_history = model_history
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.labels = labels

    def create_plot(self, model_history):
        '''Check the logloss of both train and validation, make sure they are close and have plateau'''
        plot_path = f'./plots/2DCNN-{extra_name}{data_feature}'
        plt.plot(model_history['loss'])
        plt.plot(model_history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(plot_path + '-loss.png')
        plt.close()

        plt.plot(model_history['acc'])
        plt.plot(model_history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.savefig(plot_path + '-acc.png')

    def create_results(self, model):
        '''predict on test set and get accuracy results'''
        opt = optimizers.Adam(0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])
        score = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    def confusion_results(self, X_test, y_test, labels, model):
        '''plot confusion matrix results'''
        preds = model.predict(X_test,
                              batch_size=16,
                              verbose=2)
        preds = preds.argmax(axis=1)
        preds = preds.astype(int).flatten()
        preds = (lb.inverse_transform((preds)))

        actual = y_test.argmax(axis=1)
        actual = actual.astype(int).flatten()
        actual = (lb.inverse_transform((actual)))

        classes = labels
        classes.sort()

        c = confusion_matrix(actual, preds)
        print_confusion_matrix(c, class_names=classes)

    def accuracy_results_gender(self, X_test, y_test, labels, model):
        '''Print out the accuracy score and confusion matrix heat map of the Gender classification results'''

        preds = model.predict(X_test,
                              batch_size=16,
                              verbose=2)
        preds = preds.argmax(axis=1)
        preds = preds.astype(int).flatten()
        preds = (lb.inverse_transform((preds)))

        actual = y_test.argmax(axis=1)
        actual = actual.astype(int).flatten()
        actual = (lb.inverse_transform((actual)))

        # print(accuracy_score(actual, preds))

        actual = pd.DataFrame(actual).replace({'female_angry': 'female', 'female_disgust': 'female', 'female_fear': 'female', 'female_happy': 'female', 'female_sad': 'female', 'female_neutral': 'female', 'male_angry': 'male', 'male_fear': 'male', 'male_happy': 'male', 'male_sad': 'male', 'male_neutral': 'male', 'male_disgust': 'male'
                                               })

        preds = pd.DataFrame(preds).replace({'female_angry': 'female', 'female_disgust': 'female', 'female_fear': 'female', 'female_happy': 'female', 'female_sad': 'female', 'female_surprise': 'female', 'female_neutral': 'female', 'male_angry': 'male', 'male_fear': 'male', 'male_happy': 'male', 'male_sad': 'male', 'male_surprise': 'male', 'male_neutral': 'male', 'male_disgust': 'male'
                                             })

        classes = actual.loc[:, 0].unique()
        classes.sort()

        c = confusion_matrix(actual, preds)
        print(accuracy_score(actual, preds))
        print_confusion_matrix(c, class_names=classes)


if __name__ == '__main__':
    # load dataset
    mfcc = np.load(f'./processed_data/{extra_name}{data_feature}.npy')
    ref = pd.read_csv("./metadata/Data_path.csv")
    if only_radvess:
        ref = ref.loc[ref['source'] == 'RAVDESS']

    # Split between train and test
    X_train, X_test, y_train, y_test = train_test_split(mfcc, ref.labels, test_size=0.25, shuffle=True, random_state=42
                                                        )

    # one hot encode the target
    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(y_train))
    y_test = to_categorical(lb.fit_transform(y_test))

    # Normalization as per the standard NN process
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std

    # Load model and history
    model = models.load_model(f'./models/2DCNN-{extra_name}{data_feature}')
    hist_path = f'./histories/2DCNN-{extra_name}{data_feature}'
    with open(hist_path, 'rb') as file_pi:
        model_history = pickle.load(file_pi)
    if not os.path.exists('./plots'):
        os.mkdir('./plots')

    results = get_results(model_history, model, X_test,
                          y_test, ref.labels.unique())
    results.create_plot(model_history)
    results.create_results(model)
    results.confusion_results(X_test, y_test, ref.labels.unique(), model)
