import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Activation, Dropout

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import json

from config import *


class Model:
    def __init__(self):
        self.model = None
        self.history = None
        self.X_test = None
        self.y_test = None

        with open('NN_config.json', 'r') as f:
            configuration = json.load(f)

        self.model_config = configuration["model"]
        self.training_config = configuration["training"]
        self.data_config = configuration["data"]

    def load_model(self):
        json_file = open('models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("models/model.h5")
        print("Loaded model from disk")
        opt = tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate'])
        self.model.compile(loss=self.model_config["loss"], optimizer=opt)

    def build_model(self):
        self.model = Sequential()
        for layer in self.model_config['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation, input_shape=(input_dim,)))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        opt = tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate'])
        self.model.compile(loss=self.model_config["loss"], optimizer=opt)

    def train_model(self):
        df = pd.read_csv(self.data_config["filepath"] + self.data_config["filename"], index_col=0)

        X = df[df.columns[:-1]].values
        y = df[df.columns[-1]].values

        print(X.shape, y.shape)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y,
                                                            test_size=self.data_config['train_test_split'],
                                                            random_state=0)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp,
                                                            test_size=self.data_config['train_test_split'],
                                                            random_state=0)
        self.X_test = X_test
        self.y_test = y_test

        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=self.data_config["num_classes"])
        y_valid_onehot = tf.keras.utils.to_categorical(y_valid, num_classes=self.data_config["num_classes"])

        self.history = self.model.fit(X_train, y_train_onehot, epochs=self.training_config["epochs"],
                                      validation_data=(X_valid, y_valid_onehot))

    def plot_loss(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(self.history.history["loss"], label="Training Loss")
        ax.plot(self.history.history["val_loss"], color="orange",
                label="Val loss")
        ax.legend()
        ax.set_yscale("log")
        plt.savefig("models/loss.png")
        plt.show()

    def testing_model(self):
        y_predicted = self.model.predict(self.X_test)
        y_pred_labels = np.argmax(y_predicted, axis=1)
        y_true_labels = self.y_test

        accuracy = accuracy_score(y_true_labels, y_pred_labels)

        print("Accuracy on test data: {:.2%}".format(accuracy))

        precision = precision_score(y_true_labels, y_pred_labels, average='macro')

        print("Precision on test data: {:.2%}".format(precision))

        f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
        print("F1-score on test data: {:.2%}".format(f1))

        f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
        print("F1-score on test data: {:.2%}".format(f1))

        fig = plt.figure(figsize=(10, 5))
        classification = ["disk", "old disk", "ellipsoid"]
        # creating the bar plot
        counts_predicted = np.bincount(y_pred_labels)
        counts_true = np.bincount(y_true_labels)

        plt.bar(classification, counts_true, color='blue',
                width=0.4, alpha=0.2, label="True")
        plt.bar(classification, counts_predicted, color='red',
                width=0.4, alpha=0.3, label="Predicted")

        plt.legend()
        plt.ylabel("No. of particles")
        plt.savefig("models/n_particles_class_validation_data.png")
        plt.show()

    def save_model(self):
        model_json = self.model.to_json()

        with open(self.model_config["filepath"] + "model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(self.model_config["filepath"] + "model.h5")



