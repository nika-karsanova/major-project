import numpy as np

from classes import pose
from helpers import *


def classify_video(filename: str):
    """Function called when a video is submitted to be analysed via the CLI. Uses models that have been saved and loaded
    in for prediction as opposed to training one in-real-time."""
    cap = cv.VideoCapture(filename)
    detector = pose.PoseEstimator()

    p_time = 0

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        img = detector.detect_pose(image=frame, current_frame=current_frame)

        """If the key for the current frame does not exist, that means there were no pose extraction to predict 
        the frame type on. Hence, skip and move to the next frame. """
        if len(detector.model_results[current_frame]) is None:
            continue

        """Formatting the feature dataset into a 2D array for testing the model"""
        curr = np.array([detector.model_results[current_frame]])
        n_samples, nx, ny, nz = curr.shape
        curr = curr.reshape((n_samples, nx * ny * nz))

        """Calculating FPS of the analysed  video according to the actual time passed between frames """
        c_time = time.time()
        current_fps = int(1 / (c_time - p_time))
        p_time = c_time

        # annotate_video(img, clf.predict(curr))
        # annotate_video(img, svm.predict(curr), location=constants.LEFT_CORNER2, colour=constants.RED)
        annotate_video(img, current_fps, location=constants.LEFT_CORNER3)

        cv.imshow(f"Pose for {os.path.basename(filename)}", img)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
