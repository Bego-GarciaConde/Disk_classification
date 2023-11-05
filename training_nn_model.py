import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import LSTM
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import mlflow
import mlflow.keras
from mlflow.models import infer_signature

import json
from urllib.parse import urlparse
from config import *


def eval_metrics(y_true_labels, y_pred_labels):
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='macro')
    f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
    return accuracy, precision, f1


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
        self.tag = f"SCC_relu_activation_{self.training_config['epochs']}_epochs"

    def load_model(self):
        json_file = open(f'models/{self.tag}_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(f"models/{self.tag}_model.h5")
        print("Loaded model from disk")
        opt = tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate'])
        self.model.compile(loss=self.model_config["loss"], optimizer=opt)

    def build_model(self):
        self.model = Sequential()
        for layer in self.model_config['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            if layer['type']=="dense":
                self.model.add(Dense(neurons, activation=activation, input_shape=(input_dim,),
                                     kernel_regularizer=l2(self.model_config['l2_regularizer'])))
            elif layer['type']=="dropout":
                self.model.add(Dropout(self.model_config["dropout_rate"]))


        opt = tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate'])
        # self.model.compile(loss="sparse_categorical_crossentropy", optimizer=opt)#if not one-hot coding
        self.model.compile(optimizer=opt, loss=self.model_config["loss"],  metrics=['accuracy'])

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
        # Is one hot encoding appropiate in this case?
        # We can use classification of 0, 1, 2 instead of onehot since the old disk
        # has properties between young disk and ellipsoid..

        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_valid_onehot = tf.keras.utils.to_categorical(y_valid, num_classes=3)


        self.history = self.model.fit(X_train, y_train_onehot, epochs=self.training_config["epochs"],
                                      validation_data=(X_valid, y_valid_onehot),
                                      batch_size=self.training_config["batch_size"])

        # self.history = self.model.fit(X_train, y_train, epochs=self.training_config["epochs"],
        #                             validation_data=(X_valid, y_valid),
        #                           batch_size=self.training_config["batch_size"])

        y_predicted = self.model.predict(self.X_test)
        y_pred_labels = np.argmax(y_predicted, axis=1)
        y_true_labels = self.y_test

        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        print("Accuracy on test data: {:.2%}".format(accuracy))

        precision = precision_score(y_true_labels, y_pred_labels, average='macro')
        print("Precision on test data: {:.2%}".format(precision))

        f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
        print("F1-score on test data: {:.2%}".format(f1))

        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run(run_name=run_name) as mlflow_run:
            mlflow.log_params(
                {
                    "epochs": self.training_config["epochs"],
                    "learning_rate": self.model_config["learning_rate"],
                    "batch_size": self.training_config["batch_size"],
                    "l2_regularizer": self.model_config["l2_regularizer"],
                    "dropout_rate": self.model_config["dropout_rate"],
                }
            )
            # Log the final metrics
            mlflow.log_metrics(
                {
                    "train_loss": self.history.history["loss"][-1],
                    "validation_loss": self.history.history["val_loss"][-1],
                    "test_accuracy": accuracy,
                }
            )

            mlflow.keras.autolog()
            mlflow_run_id = mlflow_run.info.run_id
            print("MLFlow Run ID: ", mlflow_run_id)

        #    signature = infer_signature(X_test, y_predicted)
       #     mlflow.keras.log_model(self.model, "model", signature=signature)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.plot(self.history.history["loss"], label="Training Loss")
            ax.plot(self.history.history["val_loss"], color="orange",
                    label="Val loss")
            ax.legend()
         #   ax.set_yscale("log")
            plt.savefig(f"models/{self.tag}_loss.png")
        #    mlflow.log_figure(fig, 'loss.png')
            plt.show()

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")

            plt.plot(self.history.history["accuracy"], label='Training Accuracy')
            plt.plot(self.history.history["val_accuracy"], label='Validation Accuracy')
            plt.title('Training and validation accuracy')
            plt.legend()
            mlflow.log_figure(fig, 'accuracy.png')


            width = 0.2
            fig = plt.figure(figsize=(10, 5))
            classification = ["disk", "old disk", "ellipsoid"]
            ind = np.arange(3)

            # creating the bar plot
            counts_predicted = np.bincount(y_pred_labels)
            counts_true = np.bincount(y_true_labels)

            plt.bar(ind - 0.15, counts_true, align='edge', color='blue', width=0.3, alpha=0.3, label="True")
            plt.bar(ind + 0.15, counts_predicted, align='edge', color='red',
                    width=0.3, alpha=0.3, label="Predicted")
            plt.xticks(ind + width, ['disk', 'old disk', 'ellipsoid'])
            plt.title(f"F1: {f1:.2f}, Prec: {precision:.2f}", fontsize=10)

            plt.legend()
            plt.ylabel("No. of particles")
            mlflow.log_figure(fig, 'test.png')
            plt.savefig(f"models/{self.tag}_n_particles_class_validation_data.png")
            plt.show()

    def save_model(self):
        model_json = self.model.to_json()

        with open(self.model_config["filepath"] + f"{self.tag}_model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(self.model_config["filepath"] + f"{self.tag}_model.h5")
