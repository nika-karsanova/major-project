from trash import test_poser
from training import *
import pandas as pd
import os


def gen_test():
    labels_df = pd.read_csv(os.path.join("output/labels/csv", f"{1}.csv"))

    if 1167 in labels_df.values:
        print(0)

    # print(labels_df.loc[labels_df['frame'] == 1167, 'category'] == 'f')

    # if (labels_df.get[(labels_df['frame'] == 1167), 'category'] == 'f').item():
    #     print(True)


if __name__ == "__main__":
    process_data()
    # test_poser()
