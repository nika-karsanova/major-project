"""File that contols the setup of various modes and call functions from other modules where appropriate"""

import concurrent.futures
import os

from classes import da
from helpers import output_func
from model import testing, training
from ui import stdin_management, model_setup


def analyse_video_provided(path: str, models):
    """Sets up the requested parameters for the video analyses."""
    print(f"Would you like to plot the distribution of landmarks by landmark and by coordinate? Yes/No")
    plot = stdin_management.verify_yes_no_query()

    print(f"Would you like to save the annotated video to a separate file? Yes/No")
    writer = stdin_management.verify_yes_no_query()

    testing.classify_video(path, models, plotting=plot, to_save=writer)


def setup_data_collection(path: str):
    """Sets up parameters and calls functions to set up the feature extraction pipeline. """

    def detail_query():
        """Configures optional parameters form the user input."""
        print(f"Would you like to prepare the data for training and/or saving? Yes/No")
        data_prep = stdin_management.verify_yes_no_query()

        if data_prep:
            X, Y = save_da_data()

            print(f"Would you like to retrain models with new dataset? Yes/No")
            tt = stdin_management.verify_yes_no_query()

            if tt:
                setup_model_retraining(X, Y)

    def path_checker():
        """Verifies whether the path passed is valid. Add it to the queue and then initialised
        a ThreadPool to speed up the feature extraction."""
        isfile, isdir = os.path.isfile(path), os.path.isdir(path)

        q = []  # files in path

        if isfile and (path.endswith(".mp4") or path.endswith('mov')):
            q.append(path)

        elif isdir:
            for file in os.listdir(path):
                if file.endswith(".mp4") or file.endswith('mov'):
                    q.append(os.path.join(path, file))

        else:
            print("\nCouldn't identify path provided. Please check whether the path is valid.\n")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            executor.map(training.data_collection, q)

    print(f"What event would you like to look for? Spin, Jump, or Fall?")
    event = stdin_management.verify_event_type_query()

    if event != da.event_type and len(da.all_train_test) != 0 and len(da.all_true_labels != 0):
        print("Warning! The event type you are looking to accumulate landmarks for is different from the one set"
              " in the system. All the landmarks accumulated so far will be lost. Continue? Yes/No")

        if stdin_management.verify_yes_no_query():
            da.empty_dataset()
            da.event_type = event
            path_checker()
            detail_query()

        else:
            detail_query()

    else:
        da.event_type = event
        path_checker()
        detail_query()


def save_da_data():
    """Function to save the data accumulated in the DataAccumulator."""
    if len(da.all_train_test) and len(da.all_true_labels) >= 10:
        print(f"Would you like to save the accumulated data? Yes/No. "
              f"If yes, the landmarks so far will be removed from memory and be saved to a file. ")

        if stdin_management.verify_yes_no_query():
            filename = stdin_management.custom_filename('train and label set')

            X, Y = training.prepare_data(all_train_test=da.all_train_test,
                                         all_true_labels=da.all_true_labels,
                                         save_data=True,
                                         filename=filename)

            da.empty_dataset()

            return X, Y

    else:
        print("Not enough data in the accumulator. Leaving... ")


def setup_model_evaluation(path: str):
    """Function to evaluate the models performance."""
    res: tuple | None = model_setup.model_eval()
    data: tuple | None = output_func.load_fvs(path)

    if res is not None and data is not None:
        X, Y = data

        print("Would you like to split the data? Yes/No")
        if stdin_management.verify_yes_no_query():
            ind = int(len(X) * 0.77)
            X, Y = X[ind:], Y[ind:]

        for m in res:
            print(f"Results for {m.__class__.__name__}")
            training.eval.labelled_data_evaluation(Y, m.predict(X))


def setup_model_retraining(data,
                           labels):
    """Sets up the parameters for retraining of the chosen models."""
    evaluate: bool = False
    print(f"Would you like to split the data into train and test for evaluation? Yes/No")
    split = stdin_management.verify_yes_no_query()

    if split:
        print(f"Would you like to evaluate retrained models? Yes/No")
        evaluate = stdin_management.verify_yes_no_query()

    print(f"Would you like to save the retrained models? Yes/No")
    save_models = stdin_management.verify_yes_no_query()

    filename = 'default'
    if save_models is True:
        filename = stdin_management.custom_filename('models')

    training.train_model(data, labels, save_models, filename, split, evaluate)


def check_path(path: str,
               func,
               models: tuple = None):
    """Function that verifies whether the provided path is valid. If it is, makes the requested call."""

    isfile: bool = os.path.isfile(path)
    isdir: bool = os.path.isdir(path)
    video: bool = (path.endswith(".mp4") or path.endswith('mov'))
    f = lambda a, b: func(a, b) if b is not None else func(a)

    if isfile and video:
        f(path, models)

    elif isdir:
        for file in os.listdir(path):
            if file.endswith('mp4') or file.endswith('mov'):
                f(os.path.join(path, file), models)

    else:
        print("\nCouldn't identify path provided. Please check whether the path is valid.\n")
