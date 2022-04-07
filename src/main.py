from pose_estimation import mediapipe_blazepose_pe
from labelling import label_videos
from training import get_labels
import sys
import numpy as np
import pandas

if __name__ == "__main__":
    get_labels("C:/Users/welleron/Desktop/mmp/github/repositories/major-project/src/output/labels/csv/1.csv")
    # label_videos()
    # mediapipe_blazepose_pe()
