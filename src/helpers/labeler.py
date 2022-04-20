import sys
import time

import cv2 as cv
import os
from helpers import assist_func, constants


def label_videos(filename: str):
    cap = cv.VideoCapture(filename)

    data = []

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        assist_func.annotate_video(frame, current_frame)
        assist_func.annotate_video(frame, int(cap.get(cv.CAP_PROP_FRAME_COUNT)), constants.RED, constants.LEFT_CORNER2)

        cv.imshow(f"Labelling video {os.path.basename(filename)}", frame)

        k = cv.waitKey(0) & 0xFF

        if k == ord("n"):
            break  # go to next video

        elif k == ord("c"):
            continue  # next frame

        elif k == ord("q"):
            sys.exit()  # quit program

        elif k == ord("s") or k == ord("f") or k == ord("z"):  # add a label to dict

            x = {
                "frame": current_frame,
                "filename": os.path.basename(filename),
                "category": chr(k),
            }

            data.append(x)

    cap.release()
    cv.destroyAllWindows()

    assist_func.output_labels(data, os.path.basename(filename)[:1])
