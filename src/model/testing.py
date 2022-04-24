import numpy as np

from classes import pose
from helpers import *
from plots import visualisations


def classify_video(filepath: str, models, plotting: bool = False, to_save: bool = False):
    """Function called when a video is submitted to be analysed via the CLI. Uses models that have been saved and loaded
    in for prediction as opposed to training one in-real-time."""
    cap = cv.VideoCapture(filepath)
    detector = pose.PoseEstimator()
    # TODO: models is a tuple, unpack into fall_clf and spin_clf (or whatever else the implementation will end up being)

    p_time = 0

    width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps: int = int(cap.get(cv.CAP_PROP_FPS))

    writer = cv.VideoWriter(os.path.join(constants.POSEDIR, f"{os.path.basename(filepath)[:-4]}.mp4v"),
                            cv.VideoWriter_fourcc("P", "I", "M", "1"),  # MPEG-1 codec used for video
                            fps,
                            (width, height),
                            isColor=False)

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

        # annotate_video(img, model.predict(curr))
        # annotate_video(img, svm.predict(curr), location=constants.LEFT_CORNER2, colour=constants.RED)
        annotate_video(img, current_fps, location=constants.LEFT_CORNER3)

        if to_save:
            writer.write(img)

        cv.imshow(f"Pose for {os.path.basename(filepath)}", img)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cap.release()
    cv.destroyAllWindows()

    if plotting:
        visualisations.process_data(detector.video_results)
