import numpy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from random import randint

from numpy import ndarray
from pandas import DataFrame

sns.set_theme(style="darkgrid",
              palette="deep",
              color_codes=True)

# plt.style.use("seaborn")


def slice_df(df: DataFrame, window: int = 500) -> DataFrame:
    grouped = df.groupby(df.index // window)

    for g in grouped:
        temp_df: DataFrame = g[1]
        yield temp_df


def plot_correlation(df: DataFrame = None, name: str = None) -> None:
    """
    Calculates the correlation between the landmarks (e.g., correlation between left eye and right eye)

    :param df: the DataFrame to calculate correlation for
    :param name: the title to give to the plot
    :return:
    """
    # print(f"Following is a Correlation Matrix for {name} \n"
    #       f"{df.corr()} \n")

    # fig, ax = plt.subplots()
    # ax.matshow(df.corr())
    # ax.set(title=f"Correlation Matrix {name}")
    # savepath = f"C:\\Users\\welleron\\Desktop\\uni_misc\\lovelace_colloq\\poster_materials\\"
    # fname = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # fig.savefig(transparent=True, fname=savepath + fname + f".png")
    # fig.show()
    # plt.close()


def multiline_plot(title: str = None, fps: int = None, **kwargs: [str, DataFrame]):
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

            savepath = f"C:\\Users\\welleron\\Desktop\\uni_misc\\lovelace_colloq\\poster_materials\\"
            fname = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + "_" + str(randint(0, 100))
            fig.show()
            fig.savefig(transparent=True, fname=savepath + fname + f".png", dpi=fig.dpi)
            # plt.close()


def plot_xyz_landmarks(df: DataFrame = None, fps: int = None) -> None:
    """
    Plots 33 graphs (one per landmark) with each of the 3 coordinates x-y-z
    as a hue on the graph over time

    :param df: DataFrame to process for plotting
    :param fps: fps in the original video
    :return:
    """

    data_x, data_y, data_z = [], [], []

    for x in df.columns:
        for i, l in enumerate(df[x]):
            if l is not None:
                data_x.append(l.x)
                data_y.append(l.y)
                data_z.append(l.z)
            else:
                data_x.append(None)
                data_y.append(None)
                data_z.append(None)

        df_xyz = pd.DataFrame(zip(data_x, data_y, data_z), columns=["x", "y", "z"])

        multiline_plot(title=f"Pose Landmark {x}", fps=fps, data=df_xyz)

        data_x.clear()
        data_y.clear()
        data_z.clear()


def plot_coordinate_over_landmarks(df: DataFrame = None, name: str = None, fps: int = None) -> None:
    """
    Plots one out of x/y/z coordinates over time for all 33 landmarks one one graph

    :param fps: FPS of the original video
    :param df: DataFrame that contains the landmark information in the format of frame -> landmark 0 ... landmark n
    :param name: given name for outputted plots
    :return:
    """

    df_x = df.applymap(lambda e: e.x if e is not None else np.nan)  # e.g., x coordinates of all landmarks over time
    df_y = df.applymap(lambda e: e.y if e is not None else np.nan)
    df_z = df.applymap(lambda e: e.z if e is not None else np.nan)

    multiline_plot(fps=fps, x_coordinate=df_x, y_coordinate=df_y, z_coordinate=df_z)

    plot_correlation(df_x, name="X")
    plot_correlation(df_y, name="Y")
    plot_correlation(df_z, name="Z")


def process_data(data: dict, fps: int = None, avg_face: bool = True,
                 avg_hands: bool = True) -> None:
    """
    Process the pose estimator output into a DataFrame for Data Analysis

    :param avg_hands: if True, instead of showing all hand landmarks, find average
    :param avg_face: if True, instead of showing all face landmarks, find average
    :param data: raw data with Landmark containers
    :param fps: FPS in the video to format the graphs with time stamp instead of frame stamp
    :return:
    """
    hashmap = {key: [] for key in range(33)}

    for frame, landmarks in data.items():
        if landmarks is not None:
            for i, l in enumerate(landmarks):  # i is index of joint, e.g. 0 for nose
                hashmap[i].append(l)  # e.g., 0 -> [[x,y,z], [x,y,z]]
        else:
            for k in hashmap.keys():
                hashmap[k].append(None)

    df = pd.DataFrame(hashmap)
    df.index = data.keys()  # making index start from one to mimic frame count
    df.index.name, df.columns.name = "frame", "landmark"

    if avg_face:
        df = df.drop(df.iloc[:, 1: 11].columns, axis=1)  # drop face landmark columns apart from nose

    if avg_hands:
        df = df.drop(df.loc[:, '17':'22'].columns, axis=1)  # drop hands landmarks columns apart from wrists

    # df.applymap(lambda e: np.array([e.x, e.y, e.z]) if e is not None else np.nan)
    # df_x = df.applymap(lambda e: e.x if e is not None else np.nan)
    # falls = df_x.to_numpy()

    # print(falls)
    #
    # is_fall = np.ones((len(falls),), dtype=int)
    #
    # return falls, is_fall

    # plot_coordinate_over_landmarks(df)
    plot_xyz_landmarks(df)
