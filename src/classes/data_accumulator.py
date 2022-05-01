"""Data Accumulator Class."""

import threading


class DataAccumulator(object):
    """Class that is used for accumulation of pose landmarks in the instance of the program.
     Utilises singleton-like approach together with class instances to prevent creation of
     variables that would make tracking of current landmarks in memory problematic. """
    __instance: object = None
    __lock: threading.Lock = threading.Lock()

    all_true_labels: list = []
    all_train_test: list = []
    event_type: str = 'f'

    def __new__(cls, *args, **kwargs):
        """Function to initialise the instance upon first creation. Ensures that only one instance exists in memory
        despite use of multithreading elsewhere in the program."""
        with cls.__lock:  # ensuring thread safety
            if not DataAccumulator.__instance:
                DataAccumulator.__instance = object.__new__(cls)
            return DataAccumulator.__instance

    @classmethod
    def empty_dataset(cls):
        """Class method to empty the accumulated landmarks upon saving or change of event the user is training models
        for. Reassigns the class instances back to empty lists."""
        cls.all_train_test = []
        cls.all_true_labels = []


da = DataAccumulator()  # creation of the Data Accumulator instances that gets imported throughout the program
