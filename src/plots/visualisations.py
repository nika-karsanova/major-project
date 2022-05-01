"""Visualisations class. Plots graphs for a video analysed in-real-time."""

import datetime
import os
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

from helpers import constants

sns.set_theme(style="darkgrid",
              palette="deep",
              color_codes=True)


# plt.style.use("seaborn")


def slice_df(df: DataFrame,
             window: int = 500) -> DataFrame:
    """If a DataFrame contains a lot of rows, slices it for plotting for better reading of the data."""
    grouped = df.groupby(df.index // window)

    for g in grouped:
        temp_df: DataFrame = g[1]
        yield temp_df


def plot_correlation(df: DataFrame,
                     videoname: str,
                     name: str = None) -> None:
    """
    Calculates the correlation between the landmarks (e.g., correlation between left eye and right eye)

    :param df: the DataFrame to calculate correlation for
    :param name: the title to give to the plot
    :return:
    """
    print(f"Following is a Correlation Matrix for {name} \n"
          f"{df.corr()} \n")

    fig, ax = plt.subplots()
    ax.matshow(df.corr())
    ax.set(title=f"Correlation Matrix {name}")

    savepath = os.path.join(constants.GRAPHDIR, f"{videoname}", "plot_correlation/")
    os.makedirs(savepath, exist_ok=True)
    fname = datetime.datetime.now().strftime("%y%m%d_%H%M%S_") + name
    # fname = "plot_correlation" + name
    fig.show()
    fig.savefig(transparent=True, fname=savepath + fname + f".png")
    # plt.close()


def multiline_plot(videoname: str,
                   title: str = '',
                   **kwargs: [str, DataFrame]):
    """
    Plotting a multiline graph for given DataFrame

    :param fps: FPS of the original video
    :param title: if provided, used as plot title. Else, the key is used
    :param kwargs: key-value pairs of DataFrames to plot
    :return:
    """

    for k, vv in kwargs.items():
        for v in slice_df(vv):

            fig, ax = plt.subplots()

            for column in v.columns:
                ax.plot(v.index, v[column], label=column)

            ax.set(ylabel="values", title=f"Title: {k}")

            if title:
                ax.set(title=f"Title: {title}")

            ax.legend(fontsize="xx-small",
                      title_fontsize="xx-small",
                      loc="center right",
                      ncol=1,
                      bbox_to_anchor=(1.11, 0.5),
                      )

            savepath: str = os.path.join(constants.GRAPHDIR, f"{videoname}",
                                         f"multiline_plots/{title if len(title) != 0 else k}/")
            os.makedirs(savepath, exist_ok=True)
            fname: str = datetime.datetime.now().strftime("%y%m%d_%H%M%S_") + str(randint(0, 1000))
            # fname = str(ax.get_title)
            fig.show()
            fig.savefig(transparent=True, fname=savepath + fname + f".png", dpi=fig.dpi)
            # plt.close()


def plot_xyz_landmarks(videoname: str,
                       df: DataFrame) -> None:
    """
    Plots 33 graphs (one per landmark) with each of the 3 coordinates x-y-z
    as a hue on the graph over time

    :param df: DataFrame to process for plotting
    :param videoname: str to use further down in creation of folders and filenames
    :return:
    """

    data_x: list = []
    data_y: list = []
    data_z: list = []

    for x in df.columns:
        for i, l in enumerate(df[x]):
            if l is not np.nan:
                data_x.append(l.x)
                data_y.append(l.y)
                data_z.append(l.z)
            else:
                data_x.append(np.nan)
                data_y.append(np.nan)
                data_z.append(np.nan)

        df_xyz: pd.DataFrame = pd.DataFrame(zip(data_x, data_y, data_z), columns=["x", "y", "z"])

        multiline_plot(videoname=videoname, title=f"pose_landmark_{x}", data=df_xyz)

        data_x.clear()
        data_y.clear()
        data_z.clear()


def plot_coordinate_over_landmarks(videoname: str,
                                   df: DataFrame) -> None:
    """
    Plots one out of x/y/z coordinates over time for all 33 landmarks one one graph

    :param df: DataFrame that contains the landmark information in the format of frame -> landmark 0 ... landmark n
    :param videoname: str to use further down in creation of folders and filenames
    :return:
    """

    df_x = df.applymap(lambda e: e.x if e is not np.nan else np.nan)  # e.g., x coordinates of all landmarks over time
    df_y = df.applymap(lambda e: e.y if e is not np.nan else np.nan)
    df_z = df.applymap(lambda e: e.z if e is not np.nan else np.nan)

    multiline_plot(videoname=videoname, x_coordinate=df_x, y_coordinate=df_y, z_coordinate=df_z)

    plot_correlation(df=df_x, name="X", videoname=videoname)
    plot_correlation(df=df_y, name="Y", videoname=videoname)
    plot_correlation(df=df_z, name="Z", videoname=videoname)


def process_data(data: dict,
                 videoname: str,
                 avg_face: bool = True,
                 avg_hands: bool = True) -> None:
    """
    Process the pose estimator output into a DataFrame for Data Analysis

    :param avg_hands: if True, instead of showing all hand landmarks, find average
    :param avg_face: if True, instead of showing all face landmarks, find average
    :param data: raw data with Landmark containers
    :param videoname: str to use further down in creation of folders and filenames
    :return:
    """
    hashmap: dict = {key: [] for key in range(33)}

    for frame, landmarks in data.items():
        if landmarks is not np.nan:
            for i, l in enumerate(landmarks):  # i is index of joint, e.g. 0 for nose
                hashmap[i].append(l)  # e.g., 0 -> [[x,y,z], [x,y,z]]
        else:
            for k in hashmap.keys():
                hashmap[k].append(np.nan)

    df: pd.DataFrame = pd.DataFrame(hashmap)
    df.index = data.keys()  # making index start from one to mimic frame count
    df.index.name, df.columns.name = "frame", "landmark"

    if avg_face:
        df = df.drop(df.iloc[:, 1: 11].columns, axis=1)  # drop face landmark columns apart from nose

    if avg_hands:
        df = df.drop(df.loc[:, '17':'22'].columns, axis=1)  # drop hands landmarks columns apart from wrists

    plot_coordinate_over_landmarks(df=df, videoname=videoname)
    plot_xyz_landmarks(df=df, videoname=videoname)
