"""Contains assist functions used throughout the projects for various applications. For example, output of labels from
the labeler, persistence of the ml and such. """

import os

import cv2 as cv
import pandas as pd

from helpers import constants
import pickle
import traceback
import numpy as np


def save_labels_csv(data, video, overwrite=False) -> None:
    """Function to output the labels one can generate through the utilization of the labeler class."""
    filename = f"{video}.csv"

    os.makedirs(constants.LABDIR, exist_ok=True)
    outfile = os.path.join(constants.LABDIR, filename)

    labels_df = pd.DataFrame(data=data)

    if not labels_df.empty and (not os.path.isfile(outfile) or overwrite):
        labels_df.to_csv(outfile, index=False)


def annotate_video(image, text, colour=constants.BLACK, location=constants.LEFT_CORNER1):
    """Function to annotate videos in OpenCV, makes use of constants for colours and locations initialized in
    constants.py file. """
    cv.putText(image,  # frame
               str(text),  # actual text
               location,  # left corner
               cv.FONT_HERSHEY_SIMPLEX,
               1.0,  # size
               colour,  # colour
               2,  # thickness
               cv.LINE_AA)


def save_model(model, filename: str):
    """Function to save the classifier used for action recognition."""

    try:
        pickle.dump(model, open(filename, 'wb'))

    except:
        print(f"Error occurred while saving model to {filename}. Printing traceback... \n")
        traceback.print_exc()


def load_model(filename: str):
    """Function to load the classifier used for action recognition."""

    try:
        return pickle.load(open(filename, 'rb'))

    except:
        print(f"Error occurred while loading a model from {filename}. Printing traceback... \n")
        traceback.print_exc()


def save_fvs(fvs, labels, filename: str):
    try:
        store = {'fvs': fvs, 'labels': labels}

        pickle.dump(store, open(filename, 'wb'))

    except:
        print(f"Error occurred while saving training data to {filename}. Printing traceback...")
        traceback.print_exc()


def load_fvs(filename: str):
    try:
        return pickle.load(open(filename, 'rb'))['fvs'], pickle.load(open(filename, 'rb'))['labels']

    except:
        print(f"Error occurred while loading train data and labels"
              f" from {filename}. Printing traceback... \n")
        traceback.print_exc()
