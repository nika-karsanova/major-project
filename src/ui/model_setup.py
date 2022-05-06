"""File to deal with loading the models used throughout the program."""

from helpers import output_func, constants
from ui import stdin_management


def model_eval():
    print(f"Model for what event type would you like to evaluate? Jump/Spin/Fall")

    to_return = []
    namings = {'j': 'jumps', 's': 'spins', 'f': 'falls'}
    event = stdin_management.verify_event_type_query()

    for m in [f'output/ml/models/{namings[event]}_clf.pkl',
              f'output/ml/models/{namings[event]}_svc.pkl',
              f'output/ml/models/{namings[event]}_nb.pkl']:

        to_return.append(output_func.load_model(m))

    return tuple(to_return)


def model_choice():
    """Loads models of choice from the persistent storage. """

    def model_config(event: str):
        """Allows user to configure the model they want to use for each individual type of classification."""

        valid = {1: f'output/ml/models/{event}_clf.pkl',
                 2: f'output/ml/models/{event}_svc.pkl',
                 3: f'output/ml/models/{event}_nb.pkl'}

        while True:
            answer = int(input("Enter your choice: "))
            if answer in valid:
                return valid[answer]
            else:
                print(f"Option out of bounds. Choose one of the given options. \n")

    to_return = []

    print(f"Load default models for classification? Yes/No")

    if stdin_management.verify_yes_no_query():
        fall_clf = output_func.load_model(constants.FALL_CLF)
        spin_clf = output_func.load_model(constants.SPIN_CLF)
        jump_clf = output_func.load_model(constants.JUMP_CLF)
        return fall_clf, spin_clf, jump_clf

    else:

        for clf in ['fall', 'spin', 'jump']:
            print("\n".join([f"Choose a model to load for {clf} detection: ",
                             f"1 -- RandomForestClassifier",
                             f"2 -- Support Vector Classifier",
                             f"3 -- Naive Bayes"]))

            event_clf = output_func.load_model(model_config(f"{clf}" + "s"))
            to_return.append(event_clf)

        return tuple(to_return)
