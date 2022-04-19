import sys
import time

import cv2 as cv
import os
from helpers import assist_func, constants

pathdir = "C:/Users/welleron/Desktop/mmp/datasets/fsv/videos/"


def label_videos():
    for video in range(17, 21):

        video_file = f"{video}.mp4"
        cap = cv.VideoCapture(os.path.join(pathdir, video_file))

        p_time = 0
        data = []

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))

            c_time = time.time()
            current_fps = int(1 / (c_time - p_time))
            p_time = c_time

            assist_func.annotate_video(frame, current_frame)
            assist_func.annotate_video(frame, current_fps, constants.BLUE, constants.LEFT_CORNER2)
            assist_func.annotate_video(frame, int(cap.get(cv.CAP_PROP_FRAME_COUNT)), constants.RED, constants.LEFT_CORNER3)

            cv.imshow(f"Labelling video {video_file}", frame)

            k = cv.waitKey(0) & 0xFF

            if k == ord("n"):
                break
            elif k == ord("c"):
                continue
            elif k == ord("q"):
                sys.exit()
            elif k == ord("s") or k == ord("f") or k == ord("z"):
                x = {
                    "frame": current_frame,
                    "filename": video_file,
                    "category": chr(k),
                }

                data.append(x)

        cap.release()
        cv.destroyAllWindows()
        assist_func.output_labels(data, video)
