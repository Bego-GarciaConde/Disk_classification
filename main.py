# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/03/2023
@author: B. García-Conde
"""
from config import *
from preparing_training_data import prepare_data
from training_nn_model import Model
from classify_snapshot import ClassifySnapshot


def main():
    if PREPARE_DATA == 1:
        prepare_data()

    if TRAIN_DATA == 1:
        nn = Model()
      #  nn.load_model()
        nn.build_model()
        nn.train_model()
        nn.save_model()

    if EVALUATE_SNAPSHOT == 1:
        snapshot = ClassifySnapshot(SNAPSHOT_TEST)
        snapshot.prepare_data_test()
        snapshot.disk_classification_nn()
        snapshot.plot_classification()



if __name__ == "__main__":
    main()
