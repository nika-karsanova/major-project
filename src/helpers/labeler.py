import sys
import time

import cv2 as cv
import os
from helpers import output_func, constants


def label_videos(filepath: str):
    """Function used as a labelling helper for assigning and saving labels from a video played using OpenCV into a
    CSV file """
    cap = cv.VideoCapture(filepath)

    data = []

    save = True

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        output_func.annotate_video(frame, current_frame)
        output_func.annotate_video(frame, int(cap.get(cv.CAP_PROP_FRAME_COUNT)), constants.RED, constants.LEFT_CORNER2)

        cv.imshow(f"Labelling video {os.path.basename(filepath)}", frame)

        k = cv.waitKey(0) & 0xFF

        if k == ord("q"):
            save = False
            break  # quit this video

        elif k == ord("c"):
            continue  # next frame

        elif k == ord("s") or k == ord("f") or k == ord("z") or k == ord("j"):  # add a label to dict

            x = {
                "frame": current_frame,
                "filename": os.path.basename(filepath),
                "category": chr(k),
            }

            data.append(x)

    cap.release()
    cv.destroyAllWindows()

    if save:
        output_func.save_labels_csv(data, os.path.basename(filepath)[:-4])
