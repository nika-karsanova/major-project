"""Video analysis configuration file. Sets up the functionality of the in-real-life analysis."""
import os
import time

import cv2 as cv
import numpy as np

from classes import pose_estimator
from helpers import constants, output_func
from plots import visualisations


def classify_video(filepath: str, models: tuple, plotting: bool = False, to_save: bool = False):
    """Function called when a video is submitted to be analysed via the CLI. Uses models that have been saved and loaded
    in for prediction as opposed to training one in-real-time."""
    cap: cv.VideoCapture = cv.VideoCapture(filepath)
    detector: pose_estimator.PoseEstimator = pose_estimator.PoseEstimator()

    fall_detector, spin_detector, jump_detector = models

    p_time: int = 0

    width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps: int = int(cap.get(cv.CAP_PROP_FPS))

    writer: cv.VideoWriter = cv.VideoWriter(os.path.join(constants.POSEDIR, f"{os.path.basename(filepath)[:-4]}.mp4v"),
                                            cv.VideoWriter_fourcc("P", "I", "M", "1"),  # MPEG-1 codec used for video
                                            fps,
                                            (width, height),
                                            isColor=False)

    while cap.isOpened():
        success: bool
        frame: np.ndarray

        success, frame = cap.read()

        if not success:
            break

        current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        img: np.ndarray = detector.detect_pose(image=frame, current_frame=current_frame)

        """If the key for the current frame does not exist, that means there were no pose extraction to predict 
        the frame type on. Hence, skip and move to the next frame. """
        if current_frame not in detector.model_results:
            continue

        """Formatting the feature dataset into a 2D array for testing the model"""
        curr: np.ndarray = np.array([detector.model_results[current_frame]])
        n_samples, nx, ny, nz = curr.shape
        curr = curr.reshape((n_samples, nx * ny * nz))

        """Calculating FPS of the analysed  video according to the actual time passed between frames """
        c_time: float = time.time()
        current_fps: int = int(1 / (c_time - p_time))
        p_time: float = c_time

        output_func.annotate_video(img,
                                   f"Fall? {fall_detector.predict(curr)}",
                                   colour=constants.GREEN)

        output_func.annotate_video(img,
                                   f"Spin? {spin_detector.predict(curr)}",
                                   location=constants.LEFT_CORNER2,
                                   colour=constants.RED)

        output_func.annotate_video(img,
                                   f"Jump? {jump_detector.predict(curr)}",
                                   location=constants.LEFT_CORNER3,
                                   colour=constants.BLUE)

        output_func.annotate_video(img,
                                   current_fps,
                                   location=constants.LEFT_CORNER4)

        if to_save:
            writer.write(img)

        cv.imshow(f"Pose for {os.path.basename(filepath)}", img)

        # k = cv.waitKey(0) & 0xFF
        #
        # if k == ord("q"):
        #     break  # quit this video
        #
        # elif k == ord("c"):
        #     continue

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cap.release()
    cv.destroyAllWindows()

    if plotting:
        visualisations.process_data(detector.video_results, videoname=f"{os.path.basename(filepath)[:-4]}_video_graphs")
