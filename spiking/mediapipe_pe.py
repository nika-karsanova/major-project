import os
from typing import NamedTuple, Any

import cv2 as cv
import mediapipe as mp

from plotting import process_data

pathdir = "C:/Users/welleron/Desktop/mmp/datasets/fsv/videos/"
outpath = "./output/pose/"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def mediapipe_blazepose_pe() -> None:
    """
    Sets up, runs and collects the output of the BlazePose model on the Fis-V figure skating dataset videos.

    :return:
    """
    cap = cv.VideoCapture(os.path.join(pathdir, "1.mp4"))

    width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fcount: int = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps: int = int(cap.get(cv.CAP_PROP_FPS))

    writer = cv.VideoWriter(os.path.join(outpath, "mediapipe_output.mp4v"),
                            cv.VideoWriter_fourcc("P", "I", "M", "1"),  # MPEG-1 codec used for video
                            fps,
                            (width, height),
                            isColor=False)

    results_data = {}  # dict to hold pose estimator output data per frame for further analysis
    data = []

    """
    Setting up the model with the provided configuration
    """
    with mp_pose.Pose(static_image_mode=False,  # treat the stream as video
                      model_complexity=2,  # higher complexity leads to more accurate results
                      smooth_landmarks=True,  # filters pose landmarks across different input images to reduce jitter
                      enable_segmentation=False,
                      smooth_segmentation=False,
                      min_tracking_confidence=0.7,
                      min_detection_confidence=0.7) as pose:

        while cap.isOpened():
            ret: bool
            frame: Any
            ret, frame = cap.read()  # ret - whether return of the frame was success, frame - actual frame

            if not ret:
                break

            # recolour image, as MediaPipe needs RGB and OpenCV is default BGR
            image = cv.cvtColor(frame, cv.COLOR_BGRA2RGB)
            image.flags.writeable = False  # performance save on memory

            results: NamedTuple = pose.process(image)  # make detection

            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # cause OpenCV works in BGR

            """Data contained within landmark array after successful pose extraction
            
            results.pose_landmarks - NormalisedLandmarkList Type
            results.pose_landmarks.landmark - RepeatedCompositeContainer type
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] - NormalizedLandmark Type
            mp_pose.POSE_CONNECTIONS - Dict, containing connections between the landmarks"""
            current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            results_data[current_frame] = None
            if results.pose_landmarks:  # if pose estimator derived an output
                results_data[current_frame] = results.pose_landmarks.landmark

            """
            Putting annotation onto the presented frame
            """
            cv.putText(image,  # frame
                       str(current_frame),  # actual text
                       (50, 50),  # left corner
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0,  # size
                       (255, 255, 255),  # colour
                       2,  # thickness
                       cv.LINE_AA)

            cv.putText(image,  # frame
                       str(fcount),  # actual text
                       (50, 100),  # left corner
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0,  # size
                       (0, 0, 0),  # colour
                       2,  # thickness
                       cv.LINE_AA)

            """
            Output of BlazePose in 3D plot representation
            """
            # mp_drawing.plot_landmarks(  # 3d plot representation
            #     results.pose_world_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     # elevation=0,
            #     # azimuth=0
            # )

            """
            Rendering the joints and adding them onto the current frame
            """
            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                      )

            """
            Displaying and saving the annotated frame
            """
            writer.write(image)
            cv.imshow(f"MediaPipe Pose Estimation", image)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        writer.release()
        cv.destroyAllWindows()

        process_data(results_data, fps, avg_face=True, avg_hands=True)
