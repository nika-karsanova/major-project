import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)


def fsd10_check() -> None:
    data = np.load("C:/Users/welleron/Desktop/mmp/datasets/fsd-10/train_data_25.npy")
    labels = np.load("C:/Users/welleron/Desktop/mmp/datasets/fsd-10/val_label_25.pkl", allow_pickle=True)
    labels_df = pd.DataFrame(data=labels)
    labels_df_t = labels_df.T

    print(labels_df_t.loc[labels_df_t[1] == 0])


def get_labels(filename: str):
    labels_df = pd.read_csv(filename)
    falls = labels_df.loc[labels_df['category'] == 'f']
    spins = labels_df.loc[labels_df['category'] == 's']

    for frame in falls['frame']:
        yield frame



