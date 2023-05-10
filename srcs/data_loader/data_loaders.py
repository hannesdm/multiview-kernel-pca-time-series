import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wget


def download_and_unzip(url, extract_to="."):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


def split_data(data, split_ratio=0.3):
    """
    Split the data into train and test set.
    :param data:
    :param split_ratio:
    :return:
    """
    num_total = data.shape[0]
    if isinstance(split_ratio, int):
        assert split_ratio > 0
        assert (
                split_ratio < num_total
        ), "test set size is configured to be larger than entire dataset."
        num_test = split_ratio
    else:
        num_test = int(num_total * split_ratio)
    num_train = num_total - num_test

    return data[:num_train], data[num_train:]


# https://archive.ics.uci.edu/ml/datasets/Hungarian+Chickenpox+Cases
def chickenpox_Dataloader(data_dir, split_ratio=0.2):
    name = f"{data_dir}/hungary_chickenpox.csv"
    if not os.path.exists(name):
        print("Data at the given path doesn't exist. Downloading now...")
        download_and_unzip(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00580/hungary_chickenpox.zip",
            data_dir,
        )

    data = pd.read_csv(name, header=0, sep=",")
    dataset = torch.tensor(
        data[
            [
                "BUDAPEST",
                "BARANYA",
                "BACS",
                "BEKES",
                "BORSOD",
                "CSONGRAD",
                "FEJER",
                "GYOR",
                "HAJDU",
                "HEVES",
                "JASZ",
                "KOMAROM",
                "NOGRAD",
                "PEST",
                "SOMOGY",
                "SZABOLCS",
                "TOLNA",
                "VAS",
                "VESZPREM",
                "ZALA",
            ]
        ].values.astype("float64")
    )
    # split dataset into train and test set
    train, test = split_data(data=dataset, split_ratio=split_ratio)
    return train, test


# https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set
# zip contains 'gt_2011.csv', 'gt_2012.csv', 'gt_2013.csv', 'gt_2014.csv', 'gt_2015.csv'
def gas_turbine_Dataloader(data_dir, subset_file="gt_2011.csv", split_ratio=0.2):
    name = f"{data_dir}/{subset_file}"
    if not os.path.exists(name):
        print("Data at the given path doesn't exist. Downloading now...")
        download_and_unzip(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00551/pp_gas_emission.zip",
            data_dir,
        )

    data = pd.read_csv(name, header=0, sep=",")
    dataset = torch.tensor(data.values)
    # split dataset into train and test set
    train, test = split_data(data=dataset, split_ratio=split_ratio)
    return train, test


# https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
def appliances_energy_Dataloader(data_dir, max_nr=5000):
    name = f"{data_dir}/energydata_complete.csv"
    if not os.path.exists(name):
        print("Data at the given path doesn't exist. Downloading now...")
        wget.download(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv",
            data_dir,
        )
    data = pd.read_csv(name, header=0, sep=",")
    dataset = torch.tensor(
        data[
            [
                "Appliances",
                "lights",
                "T1",
                "RH_1",
                "T2",
                "RH_2",
                "T3",
                "RH_3",
                "T4",
                "RH_4",
                "T5",
                "RH_5",
                "T6",
                "RH_6",
                "T7",
                "RH_7",
                "T8",
                "RH_8",
                "T9",
                "RH_9",
                "T_out",
                "Press_mm_hg",
                "RH_out",
                "Windspeed",
                "Visibility",
                "Tdewpoint",
                "rv1",
                "rv2",
            ]
        ].values[:max_nr]
    )

    # split dataset into train and test set
    train, test = split_data(data=dataset, split_ratio=0.2)
    return train, test


def lorenz_Dataloader(max_nr=2000, plot3d=False, s=10, r=28, b=2.667):
    dt = 0.01
    num_steps = max_nr

    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)

    # Set initial values
    xs[0], ys[0], zs[0] = (1.0, -1.0, 1.05)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s=10, r=28, b=2.667)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    dataset = np.vstack((xs, ys, zs)).T

    if plot3d is True:
        ax = plt.figure().add_subplot(projection="3d")
        ax.plot(xs, ys, zs, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")
        plt.show()

    # split dataset into train and test set
    train, test = split_data(data=torch.from_numpy(dataset), split_ratio=0.3)
    return train, test


def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


def sine(frequency=1, phase=0, min_t=0, max_t=120, sample_rate=0.1, scaler=1):
    t = [i for i in np.arange(min_t, max_t, sample_rate)]
    x = scaler * np.array([np.sin(2 * np.pi * frequency * i + phase) for i in t])
    return x


def sine_sum(
        frequency=[1, 20], phase=[0, 0], min_t=0, max_t=5, sample_rate=0.01, scaler=[1, 0.2]
):
    sines = []
    for i in range(len(frequency)):
        x = sine(
            frequency=frequency[i],
            phase=phase[i],
            min_t=min_t,
            max_t=max_t,
            sample_rate=sample_rate,
            scaler=scaler[i],
        )
        sines.append(x)
    return sum(sines)


def sine_sum_Dataloader(
        frequency=[1, 20], phase=[0, 0], min_t=0, max_t=5, sample_rate=0.01, scaler=[1, 0.2]
):
    x = sine_sum(
        frequency=frequency,
        phase=phase,
        min_t=min_t,
        max_t=max_t,
        sample_rate=sample_rate,
        scaler=scaler,
    )
    dataset = torch.from_numpy(x)

    # split dataset into train and test set
    train, test = split_data(dataset, split_ratio=0.2)
    return train, test


def sine_Dataloader(frequency=1, phase=0, min_t=0, max_t=120, sample_rate=0.1):
    x = sine(frequency, phase, min_t, max_t, sample_rate)

    dataset = torch.from_numpy(x)

    # split dataset into train and test set
    train, test = split_data(dataset, split_ratio=0.3)
    return train, test

def santafe_Dataloader(data_dir, training=True):
    """
    :param data_dir: location of the data
    :param training: True for training, False for testing
    :return: Tensors of training and test set
    """

    name_train = f"{data_dir}/santafe/lasertrain.dat"
    name_test = f"{data_dir}/santafe/laserpred.dat"
    if not os.path.exists(name_train):
        print("Data at the given path doesn't exist. Downloading now...")
        download_and_unzip(
            "https://cloud.esat.kuleuven.be/index.php/s/Jbgosg3XZWWrtEA/download/santafe.zip",
            data_dir,
        )

    train_data = torch.from_numpy(np.genfromtxt(name_train))
    test_data = torch.from_numpy(np.genfromtxt(name_test))

    return train_data, test_data
