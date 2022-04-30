import os

from classes import da
from model import testing, training
from ui import stdin_management


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
        filename: str = os.path.basename(path)[:-4]

        print(f"Would you like to prep to process the data for training? Yes/No")
        data_prep = stdin_management.verify_yes_no_query()

        if data_prep:
            print(f"Would you like to save the resulting dataset? Yes/No. "
                  f"If yes, the accumulated landmarks so far will be removed from memory and be saved to a file. ")
            save_data = stdin_management.verify_yes_no_query()

            if save_data:
                filename = stdin_management.custom_filename('train and label set')

            X, Y = training.prepare_data(all_train_test=da.all_train_test,
                                         all_true_labels=da.all_true_labels,
                                         save_data=save_data,
                                         filename=filename)

            print(f"Would you like to retrain models with new dataset? Yes/No")
            tt = stdin_management.verify_yes_no_query()

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
