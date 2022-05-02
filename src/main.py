"""Main Application."""

from helpers import output_func, labeler
from ui import model_setup, mode_init
import threading, multiprocessing
import time


def main(mode: str = '',
         path: str = ''):
    """Function to control the available modes and call the appropriate modules."""
    if mode == '' or mode == 'v':  # video analysis
        mode_init.check_path(path, mode_init.analyse_video_provided, model_setup.model_choice())

    elif mode == 'l':  # labelling
        mode_init.check_path(path, labeler.label_videos)

    elif mode == 'p':  # collect pose landmarks
        mode_init.setup_data_collection(path)

    elif mode == "t" and str(path).endswith('.pkl'):  # retrain model with a given fvs file
        res: tuple | None = output_func.load_fvs(path)

        if res is not None:
            train, labels = res
            mode_init.setup_model_retraining(train, labels)

    elif mode == 's':
        mode_init.save_da_data()

    elif mode == 'e' and str(path).endswith('.pkl'):
        mode_init.setup_model_evaluation(path)

    elif (mode == 't' or mode == 'e') and not str(path).endswith('.pkl'):
        print("\nPickle file was not provided. Provide a pickle file and repeat.\n")
        return

    else:
        print("No such mode, try again.", end='\n')
        return


if __name__ == "__main__":
    mode: str = ''
    path: str = ''

    while True:
        print(f"Welcome to the main menu of AI Figure Skating Commentator Application. What would you like to do?\n")

        print("\n".join([
            "l - label some videos. ",
            "p - collect landmarks from the videos for train data. ",
            "t - retrain available models. ",
            "v - run video analysis.",
            "e - evaluate a saved model.",
            "s - save data in memory to file.",
            "q - quit. note, that the unsaved landmarks in memory will be lost. ",
        ]))

        mode = str(input(f"\nChoose an option to do: ").lower())

        if mode == 'q':
            print("Thanks for using this program. Goodbye!")
            exit()

        if mode != 's':
            path = str(input(f"Provide path to a relevant file or directory (e.g., video): "))

        main(mode=mode, path=path)

        # Testing directory
        # C:/Users/welleron/Desktop/mmp/datasets/fsv/test/26.mp4
