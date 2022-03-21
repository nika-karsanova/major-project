import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

from pandas import DataFrame

sns.set_theme(style="darkgrid",
              palette="deep",
              color_codes=True)


def plot_correlation(df: DataFrame = None, name: str = None) -> None:
    """
    Calculates the correlation between the landmarks (e.g., correlation between left eye and right eye)

    :param df:
    :param name:
    :return:
    """
    print(f"\n"
          f"Following is a Correlation Matrix for {name}\n"
          f"\n"
          f"{df.corr()}")

    plt.matshow(df.corr())


def plot_xyz_landmarks(df: DataFrame = None, fps: int = None) -> None:
    """
    Plots 33 graphs (one per landmark) with each of the 3 coordinates x-y-z
    as a hue on the graph over time

    :param df:
    :param fps:
    :return:
    """
    if fps:
        x_range = [round(x / fps, 2) for x in df.index.values.tolist()]  # convert frame to seconds
        x_range = [str(datetime.timedelta(minutes=x)).rsplit(':', 1)[0] for x in x_range]

    data_x, data_y, data_z = [], [], []

    for x in range(33):
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
        df_xyz.plot(kind="line", title=f"Pose Landmark {x}")

        data_x.clear()
        data_y.clear()
        data_z.clear()

        plt.show()
        # plt.close()


def plot_coordinate_over_33_landmarks(df: DataFrame = None, name: str = None) -> None:
    """
    Plots one out of x/y/z coordinates over time for all 33 landmarks one one graph

    :param df: DataFrame that contains the landmark information in the format of frame -> landmark 0 ... landmark n
    :param name: given name for outputted plots
    :return:
    """

    df_x = df.applymap(lambda e: e.x if e is not None else np.nan)  # e.g., x coordinates of all 33 landmarks over time
    df_y = df.applymap(lambda e: e.y if e is not None else np.nan)
    df_z = df.applymap(lambda e: e.z if e is not None else np.nan)

    # sns.scatterplot(data=df_x,
    #              legend="brief",
    #              # dashes=False,
    #              ).set(ylabel="values",
    #                    title=f"x of 33 landmarks"
    #                    )

    # df_x.plot(kind="line",
    #           title="x over 33 landmarks",
    #           legend=False
    #           )
    #
    # df_y.plot(kind="line",
    #           title="y over 33 landmarks",
    #           legend=False
    #           )
    #
    # df_z.plot(kind="line",
    #           title="z over 33 landmarks",
    #           legend=False
    #           )
    #
    # plot_correlation(df_x, name="X")
    # plot_correlation(df_y, name="Y")
    # plot_correlation(df_z, name="Z")
    #
    # plt.legend(fontsize="xx-small",
    #            title_fontsize="xx-small",
    #            loc="center right",
    #            ncol=1,
    #            bbox_to_anchor=(1.2, 0.5),
    #            )
    #
    # plt.tight_layout()

    plt.show()


def process_data(data: dict, fps: int = None, fcount: int = None) -> None:
    """
    Process the pose estimator output into a DataFrame for Data Analysis

    :param data: raw data with Landmark containers
    :param fps: FPS in the video to format the graphs with time stamp instead of frame stamp
    :param fcount: total count of frames in the original video
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

    # plot_xyz_landmarks(df=df, fps=fps)
    # plot_coordinate_over_33_landmarks(df)
