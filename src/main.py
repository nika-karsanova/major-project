from helpers import output_func, labeler
from ui import model_setup, mode_init


def main(path: str, mode: str = None):
    """Function to control the available modes. """
    if mode is None or mode == 'v':  # video analysis
        mode_init.check_path(path, mode_init.analyse_video_provided, model_setup.model_choice())

    elif mode == 'l':  # labelling
        mode_init.check_path(path, labeler.label_videos)

    elif mode == 'p':  # collect pose landmarks
        mode_init.setup_data_collection(path)

    elif mode == "t" and path.endswith('.pkl'):  # retrain model with a given fvs file
        res = output_func.load_fvs(path)

        if res is not None:
            train, labels = res
            mode_init.setup_model_retraining(train, labels)

    elif mode == 't' and not path.endswith('.pkl'):
        print("\nPickle file was not provided. Provide a pickle file and repeat.\n")

    else:
        print("\nNo such mode, try again.\n")


if __name__ == "__main__":
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

        # main(mode='p', path="C:/Users/welleron/Desktop/mmp/datasets/fsv/test/")
