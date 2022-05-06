import unittest

from unittest.mock import patch

from main import main


def video_stub():
    return '26.mp4'


def pickle_stub():
    return 'falls_train_set.pkl'


class TestModuleMain(unittest.TestCase):
    """Module to test modes control through main function."""
    def test_invalid_mode_passed_int(self):
        """Test how main handles mode being passed as an int."""
        self.assertIsNone(main(mode=123))

    def test_invalid_mode_passed_str(self):
        """Test how main handles invalid mode passed in."""
        self.assertIsNone(main(mode='a'))

    def test_invalid_path_passed_for_model_training_int(self):
        """Test how main deals with invalid path as an int."""
        self.assertIsNone(main(mode='t', path=12345))

    def test_invalid_path_passed_for_model_training_str(self):
        """Test how main deals with invalid path. """
        self.assertIsNone(main(mode='t', path='12345'))

    @patch("main.mode_init.setup_model_evaluation")
    def test_e_mode(self, mock_mode_e):
        """Test weather evaluation mode calls correct methods."""
        main(mode='e', path=pickle_stub())
        mock_mode_e.assert_called_with(pickle_stub())

        self.assertIsNone(main(mode='e'))

    @patch("main.mode_init.save_da_data")
    def test_s_mode(self, mock_mode_s):
        """Test whether save move is called properly."""
        main(mode='s')
        mock_mode_s.assert_called()

    @patch("main.output_func.load_fvs", return_value=(1, 2))
    @patch("main.mode_init.setup_model_retraining")
    @patch("main.main")
    def test_t_mode(self, mock_main, mock_t, mock_l):
        """Test whether the retrain mode works as expected."""
        main(mode='t', path=pickle_stub())
        mock_l.assert_called_with(pickle_stub())

        mock_main.res = mock_l(pickle_stub())
        self.assertEqual(mock_l(pickle_stub()), mock_main.res)

        mock_t.assert_called_with(mock_main.res[0], mock_main.res[1])

    @patch("main.mode_init.setup_data_collection")
    def test_p_mode(self, mock_p):
        """Test whether pose (collect landmarks) mode looks as expected."""
        main(mode='p', path=video_stub())
        mock_p.assert_called_with(video_stub())

    @patch("main.labeler.label_videos")
    @patch("main.mode_init.check_path", return_value=1)
    def test_l_mode(self, mock_l, mock_arg1):
        """Test whether labelling mode works as expected."""
        main(mode='l', path=video_stub())
        mock_l.assert_called_with(video_stub(), mock_arg1)

    @patch("main.mode_init.model_setup.model_choice")
    @patch("main.mode_init.analyse_video_provided")
    @patch("main.mode_init.check_path", return_value=1)
    def test_v_mode(self, mock_v, mock_arg1, mock_arg2):
        """Test whether in-real-time analysis works as expected."""
        main(path=video_stub())
        mock_v.assert_called_with(video_stub(), mock_arg1, mock_arg2())


if __name__ == '__main__':
    unittest.main()
