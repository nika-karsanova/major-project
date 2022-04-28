import argparse
import math

from model import testing, training
import os
from helpers import output_func, constants, labeler
from classes import da
import logging
import sys


def model_choice():
    """Loads models of choice from the persistent storage. """

    to_return = []

    print(f"Load default models for classification? Yes/No")

    if verify_yes_no_query():
        fall_clf = output_func.load_model(constants.FALL_CLF)
        spin_clf = output_func.load_model(constants.SPIN_CLF)
        jump_clf = output_func.load_model(constants.JUMP_CLF)
        return fall_clf, spin_clf, jump_clf

    else:

        # print(f"Provide models from custom file? Yes/No. If no, you'll be offered to choose from pre-trained models.")
        #
        # if verify_yes_no_query():
        #     return output_func.load_model(custom_filename('model'))

        for clf in ['fall', 'spin', 'jump']:
            print("\n".join([f"Choose a model to load for {clf} detection: ",
                             f"1 -- RandomForestClassifier",
                             f"2 -- Support Vector Classifier",
                             f"3 -- Naive Bayes"]))

            event_clf = output_func.load_model(model_config(f"{clf}" + "s"))
            to_return.append(event_clf)

        return tuple(to_return)


def model_config(event: str):
    valid = {1: f'output/ml/models/{event}_clf.pkl',
             2: f'output/ml/models/{event}_svc.pkl',
             3: f'output/ml/models/{event}_nb.pkl'}

    while True:
        answer = int(input("Enter your choice: "))
        if answer in valid:
            return valid[answer]
        else:
            print(f"Option out of bounds. Choose one of the given options. \n")


def verify_yes_no_query():
    """Simple validator for yes/no style questions. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    while True:
        answer = input("Enter your choice: ").lower()
        if answer in valid:
            return valid[answer]
        else:
            print(f"Please respond with 'yes' or 'no' ('y' or 'n'). \n")


def analyse_video_provided(path: str, models):
    """Sets up the requested parameters for the video analyses."""
    print(f"Would you like to plot the distribution of landmarks by landmark and by coordinate? Yes/No")
    plot = verify_yes_no_query()

    print(f"Would you like to save the annotated video to a separate file? Yes/No")
    writer = verify_yes_no_query()

    testing.classify_video(path, models, plotting=plot, to_save=writer)


def setup_data_collection(path: str):
    """Sets up parameters and calls functions to set up the feature extraction pipeline. """

    # print(f"What event type do you want to collect feature vectors for? Fall/Spin")
    # event = verify_event_type_query()

    def detail_query():
        filename: str = os.path.basename(path)[:-4]

        print(f"Would you like to prep to process the data for training? Yes/No")
        data_prep = verify_yes_no_query()

        if data_prep:
            print(f"Would you like to save the resulting dataset? Yes/No. "
                  f"If yes, the accumulated landmarks so far will be removed from memory and be saved to a file. ")
            save_data = verify_yes_no_query()

            if save_data:
                filename = custom_filename('train and label set')

            X, Y = training.prepare_data(all_train_test=da.all_train_test,
                                         all_true_labels=da.all_true_labels,
                                         save_data=save_data,
                                         filename=filename)

            print(f"Would you like to retrain models with new dataset? Yes/No")
            tt = verify_yes_no_query()

            if tt:
                setup_model_retraining(X, Y)

    isfile, isdir, video = os.path.isfile(path), os.path.isdir(path), (path.endswith(".mp4") or path.endswith('mov'))

    if isfile and video:
        training.data_collection(path)
        detail_query()

    elif isdir:
        for idx, file in enumerate(os.listdir(path)):
            if file.endswith('mp4') or file.endswith('mov'):
                training.data_collection(os.path.join(path, file))

                if idx + 1 == len(os.listdir(path)):
                    detail_query()

    else:
        print("\nCouldn't identify path provided. Please check whether the path is valid.\n")


def setup_model_evaluation(data, labels, model):
    training.eval.labelled_data_evaluation(labels, model.predict(data))


def setup_model_retraining(data, labels):
    """Sets up the parameters for retraining of the chosen models."""
    evaluate = False
    print(f"Would you like to split the data into train and test for evaluation? Yes/No")
    split = verify_yes_no_query()

    if split:
        print(f"Would you like to evaluate retrained models? Yes/No")
        evaluate = verify_yes_no_query()

    print(f"Would you like to save the retrained models? Yes/No")
    save_models = verify_yes_no_query()

    filename = 'default'
    if save_models is True:
        filename = custom_filename('models')

    training.train_model(data, labels, save_models, filename, split, evaluate)


def check_path(path: str, func, models=None):
    """Function that verifies whether the provided path is valid. If it is, makes the requested call."""
    isfile, isdir, video = os.path.isfile(path), os.path.isdir(path), (path.endswith(".mp4") or path.endswith('mov'))
    f = lambda a, b: func(a, b) if b is not None else func(a)

    if isfile and video:
        f(path, models)

    elif isdir:
        for file in os.listdir(path):
            if file.endswith('mp4') or file.endswith('mov'):
                f(os.path.join(path, file), models)

    else:
        print("\nCouldn't identify path provided. Please check whether the path is valid.\n")


def main(path: str, mode: str = None):
    """Function to control the available modes. """
    if mode is None or mode == 'v':  # video analysis
        check_path(path, analyse_video_provided, model_choice())

    elif mode == 'l':  # labelling
        check_path(path, labeler.label_videos)

    elif mode == 'p':  # collect pose landmarks
        setup_data_collection(path)

    elif mode == "t" and path.endswith('.pkl'):  # retrain model with a given fvs file
        res = output_func.load_fvs(path)

        if res is not None:
            train, labels = res
            setup_model_retraining(train, labels)

    elif mode == 't' and not path.endswith('.pkl'):
        print("\nPickle file was not provided. Provide a pickle file and repeat.\n")

    else:
        print("\nNo such mode, try again.\n")


def verify_event_type_query():
    """Function to verify the type of event provided. """
    valid = {'s': 's', 'spin': 's', 'f': 'f', 'fall': 'f'}

    while True:
        answer = input("Enter your choice: ").lower()
        if answer in valid:
            return valid[answer]
        else:
            print(f"Please respond with fall or spin (f or s). \n")


def custom_filename(obj: str = 'data'):
    """Function for the user to provide custom filenames for fvs, models and such."""
    return input(f"Provide filename for the {obj}: ").lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-mode", help="Test labelling - l, "
                                      "collect pose landmarks from data and train a model - p, "
                                      "retrain models - t "
                                      "or run video analysis - v. Defaults to video analysis."
                                      "Don\'t forget to provide path to relevant files. ")

    parser.add_argument("-path", type=str, help="Provide a path to a file or directory with videos "
                                                "for labelling, analysis or feature extraction."
                                                "Alternatively, provide pickled numpy training data "
                                                "for model retraining.")

    args = parser.parse_args()

    # mode = args.mode
    # path = args.path

    # if not path:
    #     raise Exception("Provide path to a file or directory.")
    #
    # main(mode, path)

    while True:
        print(f"Welcome to the main menu of Event Recognition Application. What would you like to do?\n")

        print("\n".join([
            "l - label some videos. ",
            "p - collect landmarks from the videos for train data. ",
            "t - retrain available models. ",
            "v - run video analysis.",
            "q - quit. note, that the unsaved landmarks in memory will be lost. ",
        ]))

        mode = str(input(f"\nChoose an option to do: ").lower())

        if mode == 'q':
            print("Thanks for using this program. Goodbye!")
            exit()

        path = str(input(f"Provide path to a relevant file or directory (e.g., video): "))

        main(mode=mode, path=path)

        # main(mode='p', path="C:/Users/welleron/Desktop/mmp/datasets/fsv/videos/51.mp4")
