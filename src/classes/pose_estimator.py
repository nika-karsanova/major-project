"""Pose Estimator Class."""

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions import pose


class PoseEstimator:
    """Class that summarizes the functionality of the BlazePose Pose Estimator within this project.
    Initializes and processes data into Python lists and dictionaries for future use throughout the project
    to be accessed at any point when an object of class PoseEstimator is established.

    Runs with basic parameters, which can be tuned, if needed. Some of the available tuning parameters are not
    used in the program, hence, would not make a difference in the project."""

    def __init__(self, static_image_mode: bool = False,
                 model_complexity: int = 2,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = False,
                 min_tracking_confidence: float = 0.9,
                 min_detection_confidence: float = 0.9):
        """Function to initialise the model through the MediaPipe Python API, as well as the drawing tools and
        drawing styles. Furthermore, initialises a frame results tuple to store the results from the Pose Detection
        model per frame; video results dict, to make copies of results per frame (used for some of the plotting
        solutions); and model results, which are then being formatted as numpy arrays for training and testing of a
        selected model. """
        self.mp_pose: pose = mp.solutions.pose
        self.pose: pose.Pose = self.mp_pose.Pose(static_image_mode,
                                                 model_complexity,
                                                 smooth_landmarks,
                                                 enable_segmentation,
                                                 smooth_segmentation,
                                                 min_detection_confidence,
                                                 min_tracking_confidence)

        self.mp_drawing: mp.solutions.drawing_utils = mp.solutions.drawing_utils
        self.mp_styles: mp.solutions.drawing_styles = mp.solutions.drawing_styles

        self.frame_results: tuple = ()  # output of the 33 pose estimator landmarks (per frame)
        self.video_results: dict = {}  # a list of pose estimator results (per entry), arranged per frame
        self.model_results: dict = {}  # a list of results in [entry -> coordinate -> frame -> lm] format, arranged
        # by frame

    def detect_pose(self, current_frame: int, image: np.ndarray, draw: bool = True, plot: bool = False):
        """Function to detect the pose and analyse the output via extending the default framework functionality. If
        appropriate flag is passed, draws the joints and the connections between the landmarks on the provided frame.
        Then, returns the frame for further display. """
        # recolour image, as MediaPipe needs RGB and OpenCV is default BGR
        image = cv.cvtColor(image, cv.COLOR_BGRA2RGB)
        image.flags.writeable = False  # performance save on memory

        self.frame_results = self.pose.process(image)  # make detection

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # cause OpenCV works in BGR

        self.video_results[current_frame] = np.nan
        # self.model_results[current_frame] = [[[np.nan]], [[np.nan]], [[np.nan]]]
        lm_x: list = []
        lm_y: list = []
        lm_z: list = []

        """Data contained within landmark array after successful pose extraction:
    
                results.pose_landmarks - NormalisedLandmarkList Type
                results.pose_landmarks.landmark - RepeatedCompositeContainer type
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] - NormalizedLandmark Type
                mp_pose.POSE_CONNECTIONS - Dict, containing connections between the landmarks"""
        if self.frame_results.pose_landmarks:
            self.video_results[current_frame] = self.frame_results.pose_landmarks.landmark

            for lm in self.frame_results.pose_landmarks.landmark:
                lm_x.append(lm.x)
                lm_y.append(lm.y)
                lm_z.append(lm.z)

            self.model_results[current_frame] = [[lm_x], [lm_y], [lm_z]]

            """Rendering the joints and adding them onto the current frame"""
            if draw:
                self.mp_drawing.draw_landmarks(image=image,
                                               landmark_list=self.frame_results.pose_landmarks,
                                               connections=self.mp_pose.POSE_CONNECTIONS,
                                               landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
                                               )

            """Output of BlazePose in 3D plot representation"""
            if plot:
                self.mp_drawing.plot_landmarks(  # 3d plot representation
                    self.frame_results.pose_world_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    elevation=10,
                    azimuth=10,
                )

        return image
