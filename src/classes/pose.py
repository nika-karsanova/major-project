import cv2 as cv
import mediapipe as mp
import numpy as np


class PoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=2, smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=False, min_tracking_confidence=0.9, min_detection_confidence=0.9):

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode, model_complexity, smooth_landmarks, enable_segmentation,
                                      smooth_segmentation, min_detection_confidence, min_tracking_confidence)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.frame_results = ()  # output of the 33 pose estimator landmarks (per frame)
        self.video_results = {}  # a list of pose estimator results (per entry)
        self.model_results = {}  # a a list of results in the format of entry -> coordinate -> frame -> lm

    def detect_pose(self, current_frame, image, draw=True, plot=False):
        # recolour image, as MediaPipe needs RGB and OpenCV is default BGR
        image = cv.cvtColor(image, cv.COLOR_BGRA2RGB)
        image.flags.writeable = False  # performance save on memory

        self.frame_results = self.pose.process(image)  # make detection

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # cause OpenCV works in BGR

        # self.video_results[current_frame] = np.nan
        # self.model_results[current_frame] = [[[np.nan]], [[np.nan]], [[np.nan]]]
        lm_x, lm_y, lm_z = [], [], []

        """Data contained within landmark array after successful pose extraction

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
