import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classes import pose


def process_data(videofile: str, labfile: str):
    """Function that is using OpenCV and MediaPipe together with manually labelled video data to collected and format
    the features for the training and testing of the Machine Learning Models."""
    cap = cv.VideoCapture(videofile)

    detector = pose.PoseEstimator()

    csv_labels_df = pd.read_csv(labfile)
    falls_df = csv_labels_df.loc[csv_labels_df['category'] == 'f']
    spins_df = csv_labels_df.loc[csv_labels_df['category'] == 's']

    all_train_test, all_true_labels = [], []
    per_video_labels = {}

    tn = 0  # counter for number of True Negative samples
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        frame_check = csv_labels_df.loc[(csv_labels_df['frame'] == current_frame), 'category'] == 's'

        if frame_check.all():  # if current frame is labelled
            per_video_labels[current_frame] = True

        elif tn >= len(spins_df):  # if we already collected enough True Negative samples from this video
            continue  # move on

        else:  # otherwise
            per_video_labels[current_frame] = False
            tn += 1

        img = detector.detect_pose(image=frame, current_frame=current_frame)

        cv.imshow(f"Pose for {videofile}", img)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

    if len(detector.model_results) != 0:
        all_train_test.append(detector.model_results)
        all_true_labels.append(per_video_labels)
        prepare_data(all_train_test, all_true_labels)


def prepare_data(all_train_test: list, all_true_labels: list):
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


def train_model(train_set, true_labels):
    """Function that handles training of selected models to perform element classification. Most of the models
    used are accessible through the sklearn Machine Learning library. """

    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_set, true_labels)

    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm.fit(train_set, true_labels)
