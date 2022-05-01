"""File to control and validate the user input."""


def verify_yes_no_query():
    """Simple validator for yes/no style questions. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    while True:
        answer = input("Enter your choice: ").lower()
        if answer in valid:
            return valid[answer]
        else:
            print(f"Please respond with 'yes' or 'no' ('y' or 'n'). \n")


def verify_event_type_query():
    """Function to verify the type of event provided. """
    valid = {'s': 's', 'spin': 's', 'f': 'f', 'fall': 'f', 'j': 'j', 'jump': 'j'}

    while True:
        answer = input("Enter your choice: ").lower()
        if answer in valid:
            return valid[answer]
        else:
            print(f"Please respond with fall, spin or jump (f, s, j). \n")


def custom_filename(obj: str = 'data'):
    """Function for the user to provide custom filenames for fvs, models and such."""
    return input(f"Provide filename for the {obj}: ").lower()
