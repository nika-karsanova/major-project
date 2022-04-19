import cv2 as cv
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classes import pose
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC

from helpers import constants


def process_data(video: str):
    pathdir, labdir = "C:/Users/welleron/Desktop/mmp/datasets/fsv/videos/", "output/labels/csv/"

    train_test, is_fall = [], []

    video_file, labels_file = f"{video}.mp4", f"{video}.csv"

    cap = cv.VideoCapture(os.path.join(pathdir, video_file))
    labels_df = pd.read_csv(os.path.join(labdir, labels_file))

    detector = pose.PoseEstimator()

    labels = {}

    tn = 0
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        if current_frame in labels_df['frame'].values:
            if (labels_df.loc[(labels_df['frame'] == current_frame), 'category'] == 's').any():
                labels[current_frame] = True
            else:
                continue

        else:
            if tn >= 100:
                continue

            labels[current_frame] = False
            tn = tn + 1

        img = detector.detect_pose(image=frame, current_frame=current_frame)

        cv.imshow(f"Pose for {video_file}", img)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

    train_test.append(detector.model_results)
    is_fall.append(labels)


def prepare_data(train_test, is_fall):
    X, Y = [], []

    for x, y in zip(train_test, is_fall):
        for k in list(y):
            if k not in x:
                y.pop(k)

        X.extend(x.values())
        Y.extend(y.values())

    X = np.array(X)
    Y = np.array(Y)

    nsamples, nx, ny, nz = X.shape
    X = X.reshape((nsamples, nx * ny * nz))

    ind = int(len(X) / 2)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X[:ind], Y[:ind])

    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm.fit(X[:ind], Y[:ind])
    tn, fp, fn, tp = confusion_matrix(Y[ind:], svm.predict(X[ind:])).ravel()

    print(" TN ", tn,
          " FP ", fp,
          " FN ", fn,
          " TP ", tp)
