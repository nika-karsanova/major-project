import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

pathdir = "C:/Users/welleron/Desktop/mmp/datasets/womens_sp/videos/"
outpath = "C:/Users/welleron/Desktop/mmp/tutorials/output/"


def main():
    cap = cv.VideoCapture(os.path.join(pathdir, "1.mp4"))

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fcount = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)

    writer = cv.VideoWriter(os.path.join(outpath, "output.mp4v"),
                            cv.VideoWriter_fourcc("P", "I", "M", "1"),
                            fps,
                            (width, height), isColor=False)

    # for frame_idx in range(int(fcount)):
    while cap.isOpened():
        ret, frame = cap.read()  # ret - whether return of the frame was success, frame - actual frame

        # fgrey = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)  # Changed frame
        # writer.write(fgrey)

        cv.imshow("Test", frame)

        if cv.waitKey(int(fps)) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    writer.release()


if __name__ == "__main__":
    main()
