import numpy as np
import cv2 as cv
import os

from classes import pose
from helpers import *


def classify_video(filename: str):
    cap = cv.VideoCapture(filename)
    detector = pose.PoseEstimator()

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        img = detector.detect_pose(image=frame, current_frame=current_frame)

        if len(detector.model_results[current_frame]) == 0:
            continue

        curr = np.array([detector.model_results[current_frame]])
        nsamples, nx, ny, nz = curr.shape
        curr = curr.reshape((nsamples, nx * ny * nz))

        # annotate_video(img, clf.predict(curr))
        # annotate_video(img, svm.predict(curr), location=constants.LEFT_CORNER2, colour=constants.RED)

        cv.imshow(f"Pose for f{os.path.basename(filename)}", img)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
