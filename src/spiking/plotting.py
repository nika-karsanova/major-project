import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")


def plot_xyz_landmarks(x=None, y=None, z=None, df=None, name=None, fps=None) -> None:
    # df = pd.DataFrame(zip(x, y, z), columns=["x", "y", "z"])
    # x_range = [round(x / fps, 2) for x in df.index.values.tolist()]  # convert frame to seconds

    # sns.lineplot(data=df,
    #              dashes=False,
    #              ).set(title=f"{name}",
    #                    xlabel="index",
    #                    ylabel="values",
    #                    # xlim=(x_range[0], x_range[-1])
    #                    )

    f_x = lambda e: e.x if e is not None else np.NAN
    f_y = lambda e: e.y if e is not None else np.NAN
    f_z = lambda e: e.z if e is not None else np.NAN

    df_x = df.applymap(f_x)
    df_y = df.applymap(f_y)
    df_z = df.applymap(f_z)

    sns.lineplot(data=df_x,
                 legend="brief",
                 dashes=False,
                 )

    plt.legend(fontsize="xx-small",
               title_fontsize="xx-small",
               loc="lower left",
               # ncol=len(df_x.columns),
               )
    plt.show()


def process_data(data, fcount=None, fps=None) -> None:
    data_x, data_y, data_z = [], [], []
    hashmap = {key: [] for key in range(33)}

    for frame, landmarks in data.items():
        if landmarks is not None:
            for i, l in enumerate(landmarks):  # i is index of joint, e.g. 0 for nose
                hashmap[i].append(l)
        else:
            for k in hashmap.keys():
                hashmap[k].append(None)

    # for landmarks in data:
    #     for i, l in enumerate(landmarks):
    #         hashmap[i].append(l)  # 0 -> [[x,y,z], [x,y,z]], где 0 - нос. индекс листа с координатами - кадр.

    # for x in range(33):  # from nose [0] to right_foot [32]
    #     for y in hashmap[x]:
    #         if y is None:
    #             data_x.append(None)
    #             data_y.append(None)
    #             data_z.append(None)
    #         else:
    #             data_x.append(y.x)
    #             data_y.append(y.y)
    #             data_z.append(y.z)
    #
    #     print(data_x)
    #     print(data_y)
    #     print(data_z)
    #     plot_xyz_landmarks(data_x,
    #                        data_y,
    #                        data_z,
    #                        name=f"Pose Landmark {x}",
    #                        fps=fps)
    #     data_x.clear()
    #     data_y.clear()
    #     data_z.clear()
    #     break

    # print(f"RAW DATA: \n {data}")
    df = pd.DataFrame(hashmap)
    df.index += 1  # making index start from zero to mimic frame count
    df.index.name, df.columns.name = "frame", "landmark"

    # df = df.dropna()

    # for i, l in enumerate(df[0]):
    #     if l is not None:
    #         print("list", i, "is", l.x)

    plot_xyz_landmarks(df=df, fps=fps)
