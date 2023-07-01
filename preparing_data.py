import numpy as np
from matplotlib import rcParams

rcParams['axes.formatter.use_mathtext'] = True
from sklearn.preprocessing import StandardScaler

from config import *
from utils import component_classification, cartesian_to_spherical
import json


def load_config():
    with open('NN_config.json', 'r') as f:
        configuration = json.load(f)
    config = configuration["data"]
    return config


def load_process_df(name):
    df = pd.read_csv(path_snapshots + f"{name}_stars_Rvir.csv", index_col=0)
    disc = pd.read_csv(path_disk_classification + f"cla_disco_{name}.csv", index_col=0)
    df = pd.merge(df, disc[["ID", "JzJc", "cos_alpha"]], on="ID")
    df = cartesian_to_spherical(df)

    values = [0, 1, 2, 3]
    scaler = StandardScaler()

    lb = datos_edades.loc[datos_edades['Snapshot'] == name, 'Lookback'].iloc[0]
    conditions = component_classification(df, lb)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    df['class'] = np.select(conditions, values)
    return df


def prepare_data():
    config = load_config()
    snapshots_data = []
    for snapshot in SNAPSHOTS_TRAINING:
        print(f"Loading {snapshot}")
        df_snapshot = load_process_df(snapshot)
        snapshots_data.append(df_snapshot)

    df = pd.concat(snapshots_data, axis=0).reset_index(drop=True)
    df = df.sample(frac=1)

    print("Separating different components")
    df_disc = df[df["class"] == 0].copy()
    df_old_disc = df[df["class"] == 1].copy()
    df_ellipsoid = df[df["class"] == 2].copy()
    df_sat = df[df["class"] == 3].copy()

    # Reduce the sample of disk, they're the dominant component in number of particles
    df_disc_2 = df_disc.sample(frac=0.2, replace=False, random_state=1)

    df_final = pd.concat([df_disc_2, df_old_disc, df_ellipsoid], axis=0)
    df_final = df_final.sample(frac=1)

    # Drop columns we don't want for training
    df_final = df_final.drop(['ID',
                              'X', 'VX', 'Y', 'VY',
                              'Phi', 'Vphi',
                              'R',
                              'AlphaH', 'FeH', 'Age',
                              'Vr',
                              'AlphaFe','cos_alpha'
                              ], axis=1)
    print("Saving data test")
    df_final.to_csv(config["filepath"] + config["filename"])
