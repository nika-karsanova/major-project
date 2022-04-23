import os.path

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classes import pose
from helpers import output_func, constants
from model import eval


def data_accumulator(event_type='f') -> None:
    """A simple accumulator for all the pose estimator results from various videos used for feature extraction for the
    purpose of training Machine Learning models in FSV dataset. """
    all_train_test, all_true_labels = [], []

    for video in range(1, 21):
        videofile = os.path.join(constants.FSVPATH, f"{video}.mp4")
        labfile = os.path.join(constants.LABDIR, f"{video}.csv")

        res = collect_data(videofile, labfile, event_type=event_type)

        if res is not None:
            video_data, video_labels = res
            all_train_test.append(video_data)
            all_true_labels.append(video_labels)

    X, Y = prepare_data(all_train_test, all_true_labels, save_fvs=True, event_type=event_type)
    ind = int(len(X) / 2)

    clf, svm = train_model(X[:ind], Y[:ind], save_models=True, event_type=event_type)

    eval.labelled_data_evaluation(Y[ind:], clf.predict(X[ind:]))
    eval.labelled_data_evaluation(Y[:ind], svm.predict(X[ind:]))


def collect_data(videofile: str, labfile: str, event_type: str = "f") -> (dict, dict):
    """Function that is using OpenCV and MediaPipe together with manually labelled video data to collected and format
    the features for the training and testing of the Machine Learning Models."""
    cap = cv.VideoCapture(videofile)

    detector = pose.PoseEstimator()

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


def prepare_data(all_train_test: list, all_true_labels: list, save_fvs=False, event_type=None):
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

    if save_fvs and event_type is not None:
        output_func.save_fvs(X,
                             Y,
                             os.path.join(constants.FVSDIR, f"{event_type}_train_data"),
                             os.path.join(constants.FVSDIR, f"{event_type}_train_labels"))

    return X, Y


def train_model(train_set, true_labels, save_models=False, event_type=None):
    """Function that handles training of selected models to perform element classification. Most of the models
    used are accessible through the sklearn Machine Learning library. """

    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_set, true_labels)

    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm.fit(train_set, true_labels)

    if save_models and event_type is not None:
        output_func.save_model(clf,
                               os.path.join(constants.MLDIR, f"{event_type}_clf"))

        output_func.save_model(svm,
                               os.path.join(constants.MLDIR, f"{event_type}_svc"))

    return clf, svm
