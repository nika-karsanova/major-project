import argparse

from model import testing, training
from classes.data_accumulator import da
import os
from classes import data_accumulator
from helpers import output_func, constants, labeler


def model_choice():
    print(f"Load default models for classification? Yes/No")

    if verify_yes_no_query():
        fall_clf = output_func.load_model(constants.FALL_CLF)
        spin_clf = output_func.load_model(constants.SPIN_CLF)
        return fall_clf, spin_clf

    else:

        print("\n".join([f"Choose a model to load for fall detection: ",
                         f"1 -- ",
                         f"2 -- ",
                         f"3 -- "]))

        print("\n".join([f"Choose a model to load for spin detection: ",
                         f"1 -- ",
                         f"2 -- ",
                         f"3 -- "]))


def verify_yes_no_query():
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    while True:
        answer = input("Enter your choice: ").lower()
        if answer in valid:
            return valid[answer]
        else:
            print(f"Please respond with 'yes' or 'no' ('y' or 'n'). \n")


def analyse_video_provided(path, models):
    print(f"Would you like to plot the distribution of landmarks by landmark and by coordinate? Yes/No")
    plot = verify_yes_no_query()

    print(f"Would you like to save the annotated video to a separate file? Yes/No")
    writer = verify_yes_no_query()

    testing.classify_video(path, models, plotting=plot, to_save=writer)


def check_path(path, func, models=None):
    isfile, isdir, video = os.path.isfile(path), os.path.isdir(path), path.endswith(".mp4")
    f = lambda a, b: func(a, b) if b is not None else func(a)

    if isfile and video:
        f(path, models)

    elif isdir:
        for file in os.listdir(path):
            if file.endswith('mp4'):
                f(os.path.join(path, file), models)

    else:
        raise FileNotFoundError("Couldn't identify path provided. Please check whether the path is valid.")


def main(path: str, mode: str = None):
    if mode is None or mode == 'va':
        check_path(path, analyse_video_provided, model_choice())

    if mode == 'labelling':
        check_path(path, labeler.label_videos)

    if mode == 'pose':
        check_path(path, classes.data_accumulator)  # needs to work with path

    if mode == "retrain" and path.endswith('.pkl'):
        train, labels = output_func.load_fvs(path)
        training.train_model(train, labels)  # add filename customization

    else:
        raise FileNotFoundError("Pickle file was not provided. Provide a pickle file and repeat.")


def verify_event_type_query():
    valid = {'s': 's', 'spin': 's', 'f': 'f', 'fall': 'f'}

    while True:
        answer = input("Enter your choice: ").lower()
        if answer in valid:
            return valid[answer]
        else:
            print(f"Please respond with fall or spin (f or s). \n")


def custom_filename(obj: str = 'data'):
    return input(f"Provide filename for the {obj}: ").lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-mode", help="Test labelling, collect pose landmarks from data, retrain models "
                                      "or run video analysis. Defaults to video analysis."
                                      "Don\'t forget to provide path to relevant files. ")

    parser.add_argument("-path", type=str, help="For example, path to a video for analysis. "
                                                "Alternatively, pickled numpy training  data.")

    args = parser.parse_args()

    mode = args.mode
    path = args.path

    # if not path:
    #     raise Exception("Provide path to a file or directory.")

    # main(mode, path)

    # all_train_test.append(1)
    # all_true_labels.append(True)
    #
    # print(all_train_test, all_true_labels)
    # all_train_test, all_true_labels = [], []
    #
    # print(all_train_test, all_true_labels)
    da.all_true_labels.append(1)
    da.all_train_test.append(2)
    da.all_true_labels.append(3)
    da.all_train_test.append(4)
    print(da.all_true_labels)
    print(da.all_train_test)

    da.empty_dataset()

    print(da)
    print(da.all_true_labels, da.all_train_test,)

    da.all_true_labels.append(5)
    da.all_train_test.append(6)

    mainsample = data_accumulator.DataAccumulator()
    print(mainsample)
    print("\n", mainsample.all_true_labels, mainsample.all_train_test)
    print(da.all_true_labels, da.all_train_test, )

    mainsample.empty_dataset()
    print( da.all_true_labels, da.all_train_test,)
    print(mainsample.all_true_labels, mainsample.all_train_test)

    mainsample.all_train_test.append(10)
    print(da.all_train_test)
    da.all_train_test = []
    print(mainsample.all_train_test)
