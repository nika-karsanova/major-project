import unittest
from classes import data_accumulator, da
from multiprocessing.dummy import Pool as ThreadPool


def first():
    return 1


def second():
    return 2


class TestDataAccumulation(unittest.TestCase):
    """Module to test Data Accumulator singleton."""
    def setUp(self) -> None:
        self.sample = data_accumulator.DataAccumulator()

    def test_single_instance_created(self):
        """Ensuring that the two object instances point to the same place in memory."""
        self.assertIs(da, self.sample)

    def test_that_singleton_is_threading_safe(self):
        """Making sure that multiprocessing will not result in creation of multiple instances of DataAccumulator. """

        def create_obj(thread: int):
            print(f"running thread {thread}...")
            obj = data_accumulator.DataAccumulator()
            self.assertIs(da, obj)

        with ThreadPool(4) as pool:
            pool.map(create_obj, range(10))

    def test_cls_all_train_test_attribute(self):
        """Ensuring that the list always is the same no matter the instance of the class.
         Testing by appending the same values."""

        da.all_train_test.append(first())
        self.sample.all_train_test.append(first())

        print(da.all_train_test, self.sample.all_train_test)

        self.assertEqual(da.all_train_test, self.sample.all_train_test)

    def test_cls_all_train_test_attribute_with_different_values(self):
        """Ensuring that the list always is the same no matter the instance of the class.
        Testing by appending different values."""

        da.all_train_test.append(first())
        self.sample.all_train_test.append(second())

        print(da.all_train_test, self.sample.all_train_test)

        self.assertEqual(da.all_train_test, self.sample.all_train_test)

    def test_cls_all_true_labels_attribute(self):
        """Ensuring that the list always is the same no matter the instance of the class.
         Testing by appending the same values."""

        da.all_true_labels.append(True)
        self.sample.all_true_labels.append(True)

        print(da.all_true_labels, self.sample.all_true_labels)

        self.assertEqual(da.all_true_labels, self.sample.all_true_labels)

    def test_cls_all_true_labels_attribute_with_different_values(self):
        """Ensuring that the list always is the same no matter the instance of the class.
         Testing by appending the same values."""

        da.all_true_labels.append(False)
        self.sample.all_true_labels.append(True)

        print(da.all_true_labels, self.sample.all_true_labels)

        self.assertEqual(da.all_true_labels, self.sample.all_true_labels)

    def test_cls_attribute_deletion(self):
        """Testing impact of removal of an element from cls attribute on different cls instances."""

        da.all_train_test.append(first())
        self.sample.all_train_test.remove(first())

        self.assertTrue(len(da.all_train_test) == 0)

    def test_cls_event_type(self):
        """Testing the event type class instance."""
        da.event_type = 'j'
        self.assertTrue(self.sample.event_type == da.event_type)

    def tearDown(self) -> None:
        self.sample.empty_dataset()


if __name__ == '__main__':
    unittest.main()
