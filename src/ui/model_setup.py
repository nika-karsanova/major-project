from helpers import output_func, constants
from ui import stdin_management


def model_choice():
    """Loads models of choice from the persistent storage. """

    to_return = []

    print(f"Load default models for classification? Yes/No")

    if stdin_management.verify_yes_no_query():
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
