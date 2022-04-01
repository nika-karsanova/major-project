import pandas as pd
import cv2 as cv
import os
import time

pathdir = "C:/Users/welleron/Desktop/mmp/datasets/womens_sp/videos/"


def label_videos():
    video_file = "1.mp4"
    cap = cv.VideoCapture(os.path.join(pathdir, video_file))

    width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fcount: int = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps: int = int(cap.get(cv.CAP_PROP_FPS))

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

        cv.putText(frame,  # frame
                   str(current_frame),  # actual text
                   (50, 50),  # left corner
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0,  # size
                   (255, 255, 255),  # colour
                   2,  # thickness
                   cv.LINE_AA)

        cv.putText(frame,  # frame
                   str(current_fps),  # actual text
                   (50, 100),  # left corner
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0,  # size
                   (0, 0, 0),  # colour
                   2,  # thickness
                   cv.LINE_AA)

        cv.imshow(f"Labelling video {video_file}", frame)

        k = cv.waitKey(0) & 0xFF

        if k == ord("q"):
            break
        elif k == ord("c"):
            continue
        elif k == ord("s") or k == ord("f") or k == ord("z"):
            x = {
                "frame": current_frame,
                "filename": video_file,
                "category": chr(k),
            }

            data.append(x)

    cap.release()
    cv.destroyAllWindows()
    output_labels(data)


def output_labels(data) -> None:
    outpath = "/output/labels/"

    os.makedirs(outpath, exist_ok=True)

    labels_df = pd.DataFrame(data=data)
    labels_df.to_json(os.path.join(outpath, "labels.json"))
    labels_df.to_csv(os.path.join(outpath, "labels.csv"), index=False)
