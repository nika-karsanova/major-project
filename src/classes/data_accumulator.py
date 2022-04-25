import threading


class DataAccumulator(object):
    """Utilising singleton-like approach to the list Data Structure for data accumulation. """
    __instance = None
    __lock = threading.Lock()

    all_true_labels = []
    all_train_test = []

    def __new__(cls, *args, **kwargs):
        with cls.__lock:  # ensuring thread safety
            if not DataAccumulator.__instance:
                DataAccumulator.__instance = object.__new__(cls)
            return DataAccumulator.__instance

    @classmethod
    def empty_dataset(cls):
        cls.all_train_test = []
        cls.all_true_labels = []


da = DataAccumulator()
