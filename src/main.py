import argparse
from model import testing, training
import os
import cv2 as cv
from classes import pose
import pandas as pd


def main(training_data=None, model_selection=None, mode_selection=None, filename=None):
    if training_data is not None:
        #  training.prepare_data(train, labels)
        pass

    if model_selection is not None:
        # load a specific model
        pass

    # load a model used by default

    if mode_selection is None or mode_selection == "video_analysis":
        testing.classify_video(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-train_data", help="Path to the training data, if want to retrain a model")
    parser.add_argument("-model_choice", help="Specify a model to load, loads default otherwise")
    parser.add_argument("-mode", help="Test labeling functionality or plotting, defaults to video analysis")
    parser.add_argument("-video", help="Path to a video for analysis")

    args = parser.parse_args()

    train_data = args.train_data
    model_choice = args.model_choice
    mode = args.mode
    video = args.video
    #
    # if not (fvs or model_choice or mode or video):
    #     raise Exception("Configure the expected functionality with arguments.")

    # main(train_data, model_choice, mode, video)

    df = pd.read_csv("output/labels/csv/1.csv")
    df2 = df.loc[(df['frame'] == 1000), 'category'] == 'z'
    tn = 0
    print(tn)
