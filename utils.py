import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from config import *


def cartesian_to_spherical(df):
    r_polar = df["X"] ** 2 + df["Y"] ** 2
    df["R_sph"] = np.sqrt(r_polar + df["Z"] ** 2)
    df["R"] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2)
    df["VR"] = (df["X"] * df["VX"] + df["Y"] * df["VY"]) / df["R"]
    return df


def component_classification(df, lb_time):
    lb_ref = 0.01321
    #  time = (13.78 - lb_time)*1000
    age_since_merger = 9500 - (lb_time - lb_ref) * 1000
    print(age_since_merger)
    conditions = [
        (df['Age'] <= age_since_merger) & (df["R_sph"] < 25),
        (df['Age'] > age_since_merger) & (df["R_sph"] < 25) & (df['cos_alpha'] > 0.7),
        (df['Age'] > age_since_merger) & (df["R_sph"] < 25) & (df['cos_alpha'] < 0.7),
        (df["R_sph"] > 25),
    ]
    return conditions


def histogram_2d(df, coord1, coord2, ax, bins=150):
    """
    :param df: data frame
    :param coord1: "X"/ "Y" /"Z"
    :param coord2: "X"/ "Y" /"Z"
    :param ax: subplot
    :param bins: number of bins for histogram

    """
    rangexy = [-25, 25]
    #        binsx=150

    aspect_r = (rangexy[1] - rangexy[0]) / (rangexy[1] - rangexy[0])
    extent_r = [rangexy[0], rangexy[1], rangexy[0], rangexy[1]]
    surface = (((rangexy[1] - rangexy[0]) / bins) ** 2) * 1e6

    stat0 = stats.binned_statistic_2d(df[coord1], df[coord2], df['Mass'] / surface, statistic='sum',
                                      bins=(bins, bins),
                                      range=[rangexy, rangexy])
    im = np.flip(stat0.statistic.T * 1., 0)
    den = ax.imshow(np.log10(im), cmap='inferno_r', extent=extent_r,
                    aspect=aspect_r, vmin=-0.5, vmax=2.5)
    return den
