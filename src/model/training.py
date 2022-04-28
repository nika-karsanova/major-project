import os.path
from typing import Any

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classes import pose, da
from helpers import output_func, constants
from model import eval


def data_collection(path: str, event_type='j'):
    """A simple accumulator for all the pose estimator results from various videos used for feature extraction for the
    purpose of training Machine Learning models in FSV dataset. """

    res = collect_data(path,
                       os.path.join(constants.LABDIR, f"{os.path.basename(path)[:-4]}.csv"),
                       event_type=event_type)

    if res is not None:
        video_data, video_labels = res
        da.all_train_test.append(video_data)
        da.all_true_labels.append(video_labels)


def collect_data(videofile: str, labfile: str, event_type: str = "f") -> (dict, dict):
    """Function that is using OpenCV and MediaPipe together with manually labelled video data to collected and format
    the features for the training and testing of the Machine Learning Models."""
    cap = cv.VideoCapture(videofile)

    detector = pose.PoseEstimator()

    if not os.path.isfile(labfile):
        print(f"No label file was found at {labfile}. Attempting to load next video... ")
        return

    csv_labels_df = pd.read_csv(labfile)
    event_df = csv_labels_df.loc[csv_labels_df['category'] == f"{event_type}"]

    per_video_labels = {}

    tn = 0  # counter for number of True Negative samples
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        frame_check = csv_labels_df.loc[(csv_labels_df['frame'] == current_frame), 'category'] == f"{event_type}"

        if frame_check.any():  # if current frame is labelled as required event type
            per_video_labels[current_frame] = True

        elif tn >= len(event_df):  # if we already collected enough True Negative samples from this video
            continue  # move on

        elif current_frame not in csv_labels_df['frame'].values:  # otherwise
            per_video_labels[current_frame] = False
            tn += 1

        else:
            continue

        img = detector.detect_pose(image=frame, current_frame=current_frame)

        cv.imshow(f"Pose for {videofile}", img)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

    if len(detector.model_results) != 0:
        return detector.model_results, per_video_labels


def prepare_data(all_train_test: list, all_true_labels: list, save_data=False, filename='default'):
    """Function to convert collected data into a 2D numpy array for training and testing ML models.
    Initially, Converts the list of dicts into 4D numpy array."""

    X, Y = [], []

    for x, y in zip(all_train_test, all_true_labels):
        for k in list(y):
            if k not in x:
                y.pop(k)

        X.extend(x.values())
        Y.extend(y.values())

    X, Y = np.array(X), np.array(Y)

    nsamples, nx, ny, nz = X.shape
    X = X.reshape((nsamples, nx * ny * nz))

    if save_data:
        output_func.save_fvs(X, Y, os.path.join(constants.FVSDIR, f"{filename}_train_set.pkl"))
        da.empty_dataset()

    return X, Y


def train_model(X, Y, save_models=False, filename='default', split=True, evaluate=True):
    """Function that handles training of selected models to perform element classification. Most of the models
    used are accessible through the sklearn Machine Learning library. """

    # Y = Y.reshape(Y.shape[0], 1)
    train_set, train_labels = X, Y
    test_set, test_labels = None, None

    if split:
        # train_set, train_labels, test_set, test_labels = train_test_split(X,
        #                                                                   Y,
        #                                                                   test_size=0.2,
        #                                                                   random_state=42,
        #                                                                   shuffle=False,
        #                                                                   stratify=None)
        ind = int(len(X) * 0.9)
        train_set, train_labels = X[:ind], Y[:ind]
        test_set, test_labels = X[ind:], Y[ind:]

    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_set, train_labels)

    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm.fit(train_set, train_labels)

    nb = GaussianNB()
    nb.fit(train_set, train_labels)

    if evaluate and (test_set, test_labels) is not None:
        eval.labelled_data_evaluation(test_labels, clf.predict(test_set))
        eval.labelled_data_evaluation(test_labels, svm.predict(test_set))
        eval.labelled_data_evaluation(test_labels, nb.predict(test_set))

    if save_models:
        output_func.save_model(clf, os.path.join(constants.MLDIR, f"{filename}_clf.pkl"))
        output_func.save_model(svm, os.path.join(constants.MLDIR, f"{filename}_svc.pkl"))
        output_func.save_model(svm, os.path.join(constants.MLDIR, f"{filename}_nb.pkl"))

    return clf, svm
