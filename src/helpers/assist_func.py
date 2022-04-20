"""Contains assist functions used throughout the projects for various applications. For example, output of labels from
the labeler, persistence of the ml and such. """

import os

import cv2 as cv
import pandas as pd

from helpers import constants


def output_labels(data, video, overwrite=False) -> None:
    """Function to output the labels one can generate through the utilization of the labeler class."""
    outpath_csv = "./output/labels/csv/"
    filename = f"{video}.csv"

    os.makedirs(outpath_csv, exist_ok=True)
    outfile = os.path.join(outpath_csv, filename)

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


def save_model():
    """Function to save the classifier used for action recognition."""
    pass


def load_model():
    """Function to load the classifier used for action recognition."""
    pass
