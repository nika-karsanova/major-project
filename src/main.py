import argparse
import sys

from model import testing, training
import os
import cv2 as cv
from classes import pose
import pandas as pd
from helpers import output_func, constants, labeler


def model_choice():
    print(f"Load default models for classification? Yes/No")

    if verify_yes_no_query():
        fall_clf = output_func.load_model(constants.FALL_CLF)
        spin_clf = output_func.load_model(constants.SPIN_CLF)
        return fall_clf, spin_clf

    else:

        print("\n".join([f"Choose a model to load for fall detection: ",
                         f"1 -- ",
                         f"2 -- ",
                         f"3 -- "]))

        print("\n".join([f"Choose a model to load for spin detection: ",
                         f"1 -- ",
                         f"2 -- ",
                         f"3 -- "]))


def verify_yes_no_query():
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    while True:
        answer = input("Enter your choice: ").lower()
        if answer in valid:
            return valid[answer]
        else:
            print(f"Please respond with 'yes' or 'no' ('y' or 'n'). \n")


def analyse_video_provided(path, models):
    print(f"Would you like to plot the distribution of landmarks by landmark and by coordinate? Yes/No")
    plot = verify_yes_no_query()

    print(f"Would you like to save the annotated video to a separate file? Yes/No")
    writer = verify_yes_no_query()

    testing.classify_video(path, models, plotting=plot, to_save=writer)


def check_path(path, func, models=None):
    isfile, isdir, video = os.path.isfile(path), os.path.isdir(path), path.endswith(".mp4")
    f = lambda a, b: func(a, b) if b is not None else func(a)

    if isfile and video:
        f(path, models)

    elif isdir:
        for file in os.listdir(path):
            if file.endswith('mp4'):
                f(os.path.join(path, file), models)

    else:
        raise FileNotFoundError("Couldn't identify path provided. Please check whether the path is valid.")


def main(path: str, mode: str = None):
    if mode is None or mode is 'va':
        check_path(path, analyse_video_provided, model_choice())

    if mode is 'labelling':
        check_path(path, labeler.label_videos)

    if mode is 'pose':
        check_path(path, training.data_accumulator)

    if mode is "retrain" and path.endswith('.pkl'):
        train, labels = output_func.load_fvs(path)
        training.train_model(train, labels)

    else:
        raise FileNotFoundError("Pickle file was not provided. Provide a pickle file and repeat.")


def custom_filename(obj: str = 'data'):
    return input(f"Provide filename for the {obj}: ").lower()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-mode", help="Test labelling, collect pose landmarks from data, retrain models "
                                      "or run video analysis. Defaults to video analysis."
                                      "Don\'t forget to provide path to relevant files. ")

    parser.add_argument("-path", type=str, help="For example, path to a video for analysis. "
                                                "Alternatively, pickled numpy training  data.")

    args = parser.parse_args()

    mode = args.mode
    path = args.path

    # if not path:
    #     raise Exception("Provide path to a file or directory.")

    # main(mode, path)
