"""Video labeler."""

import os

import cv2 as cv
import numpy as np

from helpers import output_func, constants


def label_videos(filepath: str):
    """Function used as a labelling helper for assigning and saving labels from a video played using OpenCV into a
    CSV file. Supports following labels: z - for the frames, which are zoomed in on the skaters and would not be the
    best pose samples for training; f - to identify a fall; s - to identify a spin; j - to identify a jump."""
    cap: cv.VideoCapture = cv.VideoCapture(filepath)

    data: list = []

    save: bool = True

    while cap.isOpened():
        success: bool
        frame: np.ndarray

        success, frame = cap.read()

        if not success:
            break

        current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        output_func.annotate_video(frame, current_frame)
        output_func.annotate_video(frame, int(cap.get(cv.CAP_PROP_FRAME_COUNT)), constants.RED, constants.LEFT_CORNER2)

        cv.imshow(f"Labelling video {os.path.basename(filepath)}", frame)

        k: int = cv.waitKey(0) & 0xFF  # stores the pressed value in unicode representaion

        if k == ord("q"):
            save = False
            break  # quit this video

        elif k == ord("c"):
            continue  # next frame

        elif k == ord("s") or k == ord("f") or k == ord("z") or k == ord("j"):  # add a label to dict

            x = {
                "frame": current_frame,
                "filename": os.path.basename(filepath),
                "category": chr(k),  # converts the k value in unicode to char representation
            }

            data.append(x)

    cap.release()
    cv.destroyAllWindows()

    if save:
        # does not prompt for a filename, saves in with the video filename instead
        output_func.save_labels_csv(data, os.path.basename(filepath)[:-4])
