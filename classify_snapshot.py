import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.formatter.use_mathtext'] = True
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from config import *
from utils import cartesian_to_spherical, histogram_2d
from nn_model import Model


class ClassifySnapshot:
    def __init__(self, name):
        self.name = name
        self.stars_data = pd.read_csv(path_snapshots + f"{self.name}_stars_Rvir.csv", sep=",")
        self.cla_disk_data = pd.read_csv(path_disk_classification + f"cla_disco_996.csv", index_col=0)
        self.nn_model = None
        self.test_data = None
        self.complete_data = None
        self.disk = None
        self.old_disk = None
        self.ellipsoid = None

    def prepare_data_test(self):
        df_complete = pd.merge(self.stars_data, self.cla_disk_data[["ID", "JzJc", "cos_alpha"]], on="ID")
        df_complete = cartesian_to_spherical(df_complete)

        df_complete = df_complete[df_complete["R_sph"] < 30].copy()
        self.complete_data = df_complete
        df_test = df_complete.drop({'ID',
                                    'X', 'VX', 'Y', 'VY',
                                    'Phi',
                                    'Vphi',
                                    'R',
                                    'AlphaH',
                                    'FeH',
                                    'Age',
                                    'Vr',
                                    'AlphaFe',
                                    }, axis=1)
        self.test_data = df_test

    def disk_classification_nn(self):
        nn = Model()
        nn.load_model()

        nn.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                         loss=tf.keras.losses.CategoricalCrossentropy())

        opt = tf.keras.optimizers.Adam(learning_rate=nn.model_config['learning_rate'])
        nn.model.compile(loss=nn.model_config["loss"], optimizer=opt)

        scaler = StandardScaler()
        data_test = scaler.fit_transform(self.test_data)

        y_predicted = nn.model.predict(data_test)

        y_pred_labels = np.argmax(y_predicted, axis=1)

        self.complete_data["Class"] = y_pred_labels

        self.disk = self.complete_data[0 == self.complete_data["Class"]].copy()
        self.old_disk = self.complete_data[1 == self.complete_data["Class"]].copy()
        self.ellipsoid = self.complete_data[2 == self.complete_data["Class"]].copy()

    def plot_classification(self):
        #  plt.rc(family='serif')
        fig, ax = plt.subplots(2, 3, figsize=(6, 4), sharex=True, sharey=True)

        print(1)
        den = histogram_2d(self.disk, "X", "Z", ax[0, 0])
        print(2)
        den = histogram_2d(self.disk, "X", "Y", ax[1, 0])
        print(3)
        den = histogram_2d(self.old_disk, "X", "Z", ax[0, 1])
        print(4)
        den = histogram_2d(self.old_disk, "X", "Y", ax[1, 1])
        print(5)
        den = histogram_2d(self.ellipsoid, "X", "Z", ax[0, 2])
        print(6)
        den = histogram_2d(self.ellipsoid, "X", "Y", ax[1, 2])

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)

        ax[0, 0].set_title(r"Disk", fontsize=16)
        ax[0, 1].set_title(r"Old Disk ", fontsize=16)
        ax[0, 2].set_title(r"Ellipsoid", fontsize=16)

        cbar_ax = fig.add_axes([1.01, 0., 0.025, 0.95])
        cbar = fig.colorbar(den, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=16, top=True, bottom=False,
                            labeltop=True, labelbottom=False)
        cbar.set_label(label=r"log $\Sigma_{\ast}$ [$\mathrm{M_{\odot}}$/ $\mathrm{pc^2}$]", size=18)
        cbar.ax.tick_params(labelsize=16)

        for i in range(3):
            for j in range(2):
                ax[j, i].tick_params(labelsize=16)

        ax[0, 0].set_ylabel("Z [kpc]", fontsize=18)
        ax[1, 0].set_ylabel("Y [kpc]", fontsize=18)

        plt.subplots_adjust(left=0, bottom=0., right=1, top=1, wspace=0.0, hspace=0.0)
        plt.savefig(f"results/classification_{self.name}.png", bbox_inches="tight")
        plt.show()

    def save_classified_snapshot(self):
        self.complete_data.to_csv(f"results/classified_{self.name}.csv")
