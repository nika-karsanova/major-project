# open cv a video
# load in cvs labels
# compare frame in csv to current frame
# if label is z - do not run pose estimation, pose/frame reading are NaN
# if label is f - run pose estimation, store results for individual frames as TP for falls
# if label is s - run pose estimation, store results for temporal frames as TP for spins

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


def process_data():
    pathdir = "C:/Users/welleron/Desktop/mmp/datasets/fsv/videos/"
    labdir = "output/labels/csv/"

    train_test = []
    is_fall = []

    for video in range(1, 21):
        video_file = f"{video}.mp4"
        labels_file = f"{video}.csv"

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

    X, Y = [], []

    for x, y in zip(train_test, is_fall):
        for k in list(y):
            if k not in x:
                y.pop(k)

        # print("X keys:", x.keys())
        # print("Y keys", y.keys())
        X.extend(x.values())
        Y.extend(y.values())

    # for y in is_fall:
    #     print(y.keys())
        # Y.extend(y)

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

    for video in range(21, 22):
        video_file = f"{video}.mp4"
        cap = cv.VideoCapture(os.path.join(pathdir, video_file))
        detector = pose.PoseEstimator()

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            img = detector.detect_pose(image=frame, current_frame=current_frame)

            if detector.model_results[current_frame] == 0:
                continue

            curr = np.array([detector.model_results[current_frame]])
            nsamples, nx, ny, nz = curr.shape
            curr = curr.reshape((nsamples, nx * ny * nz))
            annotate_video(img, clf.predict(curr))
            annotate_video(img, svm.predict(curr), location=constants.LEFT_CORNER2, colour=constants.RED)

            cv.imshow(f"Pose for {video_file}", img)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv.destroyAllWindows()


def annotate_video(image, text, colour=constants.BLACK, location=constants.LEFT_CORNER1):
    cv.putText(image,  # frame
               str(text),  # actual text
               location,  # left corner
               cv.FONT_HERSHEY_SIMPLEX,
               1.0,  # size
               colour,  # colour
               2,  # thickness
               cv.LINE_AA)