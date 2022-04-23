import argparse
from model import testing, training
import os
import cv2 as cv
from classes import pose
import pandas as pd
from helpers import output_func, constants, labeler


def files_to_label(path):
    if os.path.isdir(path):
        labeler.label_videos(path)

    elif os.path.isfile(path):
        for file in os.listdir(path):
            if file.endswith(".mp4"):
                labeler.label_videos(os.path.join(path, file))

    else:
        raise FileNotFoundError("Couldn't identify path provided. Please check whether the path is valid.")


def handle_non_default_modes(mode: str, path: str):
    if mode == "labelling":
        files_to_label(path)

    # elif mode == "plotting":
    #     pass

    elif mode == "pose":
        pass

    elif mode == "retrain":
        pass


def main(path: str = None, mode: str = None):
    if mode is not None:
        handle_non_default_modes(mode, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-mode", help="Test labelling, video plotting, collect pose landmarks for data, retrain models "
                                      "or run video analysis. Defaults to video analysis."
                                      "Don\'t forget to provide video to analyse. ")

    parser.add_argument("-model", help="Choose what model to load. If not provided, loads default ones. ")

    parser.add_argument("-train_data", type=str, help="Path to the training data file, if want to retrain a model. ")

    parser.add_argument("-path", type=str, help="Path to a video for analysis, or labelling and such. "
                                                "A directory of videos can be provided alternatively.")

    args = parser.parse_args()

    # mode = args.mode
    # train_data = args.train_data
    # model = args.model_choice
    # video = args.video

    # if not (train_data or model or mode or video):
    #     raise Exception("Configure the expected functionality with arguments.")

    # main(train_data, model_choice, mode, video)

    # training.data_accumulator('f')
