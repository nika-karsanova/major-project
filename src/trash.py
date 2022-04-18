import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

from classes import pose
import cv2 as cv
from helpers import constants

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)


def fsd10_check() -> None:
    data = np.load("C:/Users/welleron/Desktop/mmp/datasets/fsd-10/train_data_25.npy")
    labels = np.load("C:/Users/welleron/Desktop/mmp/datasets/fsd-10/val_label_25.pkl", allow_pickle=True)
    labels_df = pd.DataFrame(data=labels)
    labels_df_t = labels_df.T

    # print(labels_df_t.loc[labels_df_t[1] == 0])
    print(data.shape)


def get_falls(filename: str):
    labels_df = pd.read_csv(filename)
    falls = labels_df.loc[labels_df['category'] == 'f']
    spins = labels_df.loc[labels_df['category'] == 's']

    for frame in falls['frame']:
        yield frame


def get_spins(filename: str):
    labels_df = pd.read_csv(filename)
    falls = labels_df.loc[labels_df['category'] == 'f']
    spins = labels_df.loc[labels_df['category'] == 's']

    for frame in spins['frame']:
        yield frame


def test_poser():
    pathdir = "C:/Users/welleron/Desktop/mmp/datasets/fsv/videos/"
    outpath = "./output/pose/"
    tp_falls = []
    tn_falls = []

    for video in range(1, 6):
        video_file = f"{video}.mp4"
        labels_df = pd.read_csv(os.path.join("output/labels/csv", f"{video}.csv"))

        cap = cv.VideoCapture(os.path.join(pathdir, video_file))
        width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fcount: int = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps: int = int(cap.get(cv.CAP_PROP_FPS))

        detector = pose.PoseEstimator()

        for f in labels_df['frame'].loc[labels_df['category'] == 'f']:

            # writer = cv.VideoWriter(os.path.join(outpath, f"{video}.mp4v"),
            #                         cv.VideoWriter_fourcc("P", "I", "M", "1"),  # MPEG-1 codec used for video
            #                         fps,
            #                         (width, height),
            #                         isColor=False)

            # while cap.isOpened():
            cap.set(1, f - 1)  # use specific frame
            success, frame = cap.read()

            if not success:
                break

            current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            img = detector.detect_pose(image=frame, current_frame=current_frame)
            # print(detector.results.pose_landmarks.landmark) # resets every frame
            # print(detector.results_data)  # appends at every frame

            annotate_video(img, current_frame, )
            annotate_video(img, fcount, location=constants.LEFT_CORNER2)

            # writer.write(img)
            cv.imshow(f"Pose for {video_file}", img)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        if len(detector.model_results) != 0:
            tp_falls.append(detector.model_results)

        # detector2 = pose.PoseEstimator()

        # c = 0
        # while c != 30:
        #     success, frame = cap.read()
        #
        #     if not success:
        #         break
        #
        #     current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        #
        #     if current_frame in labels_df:
        #         continue
        #
        #     img = detector2.detect_pose(image=frame, current_frame=current_frame)
        #
        #     # cv.imshow(f"Pose for {video_file}", img)
        #
        #     if cv.waitKey(1) & 0xFF == ord("q"):
        #         break
        #
        #     c = c + 1
        #     # print(c)
        #
        # if len(detector2.model_results) != 0:
        #     tn_falls.append(detector2.model_results)

        """
        Training and Testing data is in a format of 4D NumPy Array, similar to the following example.
        
        X = np.array([[[[1, 1, 1, 1]], [[2, 2, 2, 2]], [[3, 3, 3, 3]]],  
                      [[[4, 4, 4, 4]], [[5, 5, 5, 5]], [[6, 6, 6, 6]]]])
                      
        It's shape is (ExCxFxL), where E is Entry, C is Coordinate, F is Frame and L is Landmark
        """

        #
        # clf = RandomForestClassifier(random_state=0)
        # clf.fit(train_data[:7], train_labels[:7])
        #
        # print(clf.predict(train_data[7:]))

        # clf = OneClassSVM(gamma='auto').fit(train_data)
        # print(clf.predict(train_data))

        # writer.release()
        cap.release()
        cv.destroyAllWindows()

    # print(len(tn_falls))
    # print(tn_falls)

    X = []

    for x in tp_falls:
        X.extend(x.values())

    X = np.array(X)
    print(X)
    print(X.shape)

    nsamples, nx, ny, nz = X.shape
    X = X.reshape((nsamples, nx * ny * nz))
    Y = np.ones(len(X), dtype=int)

    print(len(X))

    # clf = RandomForestClassifier(random_state=0)
    # clf.fit(X, Y)

    # Test per video frame
    # for video in range(1, 3):
    #     video_file = f"{video}.mp4"
    #
    #     cap = cv.VideoCapture(os.path.join(pathdir, video_file))
    #
    #     detector2 = pose.PoseEstimator()
    #
    #     while cap.isOpened():
    #         success, frame = cap.read()
    #
    #         if not success:
    #             break
    #
    #         current_frame: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    #         img = detector2.detect_pose(image=frame, current_frame=current_frame)
    #
    #         test = np.array(detector2.ml_data[current_frame])
    #         nsamples, nx, ny = test.shape
    #         test = test.reshape((nsamples, nx * ny))
    #         print(clf.predict(test))
    #
    #         cv.imshow(f"Pose for {video_file}", img)
    #
    #         if cv.waitKey(1) & 0xFF == ord("q"):
    #             break
    #
    #     cap.release()
    #     cv.destroyAllWindows()


def annotate_video(image, text, colour=constants.BLACK, location=constants.LEFT_CORNER1):
    cv.putText(image,  # frame
               str(text),  # actual text
               location,  # left corner
               cv.FONT_HERSHEY_SIMPLEX,
               1.0,  # size
               colour,  # colour
               2,  # thickness
               cv.LINE_AA)
