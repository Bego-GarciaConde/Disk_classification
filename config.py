import pandas as pd

path_datos = "/home/bego/GARROTXA_puigsacalm//datos_GARROTXA_resim/"
path_disk_classification = "/home/bego/GARROTXA_copia_paradox/disco/"
path_snapshots = "/home/bego/GARROTXA/snapshots/"

datos_edades = pd.read_csv(path_datos + "edades.csv", sep=",", index_col=0)

# Select three snapshots for the training data


# -----PREPARING DATA------
PREPARE_DATA = 0
SNAPSHOTS_TRAINING = [999, 890, 790]

# ----TRAINING DATA-------
TRAIN_DATA = 0

# ----EVALUATE SNAPSHOT----

EVALUATE_SNAPSHOT = 1
SNAPSHOT_TEST = 996
