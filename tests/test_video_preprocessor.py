import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import the module to be tested
try:
    from video_preprocessor import preprocess_target_video
except ImportError:
    preprocess_target_video = None # Placeholder if module/function doesn't exist yet

class TestVideoPreprocessor(unittest.TestCase):

    def setUp(self):
        # Setup that might be common for several tests
        self.test_video_path = "dummy_video_for_preprocessor.mp4"
        # Create a dummy file; actual video content is not strictly needed if process_video is mocked
        with open(self.test_video_path, 'w') as f:
            f.write("dummy video content")

        # Mock config values that might be used
        self.mock_config_patcher = patch.dict('os.environ', {
            'SQLALCHEMY_DATABASE_URL': 'sqlite:///./test_preprocessor.db',
            'ENABLE_CHECKSUM': 'True',
            'LOG_LEVEL': 'DEBUG', # Or directly patch config module attributes
            'FRAME_INTERVAL': '1' # Example, if process_assets uses it
        })
        self.mock_config_patcher.start()


    def tearDown(self):
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
        if os.path.exists("test_preprocessor.db"): # Clean up dummy DB
            os.remove("test_preprocessor.db")
        
        # Clean up any timestamp JSON file that might be created by a full run
        expected_json_path = f"{os.path.splitext(os.path.basename(self.test_video_path))[0]}_timestamps.json"
        if os.path.exists(expected_json_path):
            os.remove(expected_json_path)
            
        self.mock_config_patcher.stop()


    @patch('video_preprocessor.database') # Mock the entire database module used by video_preprocessor
    @patch('video_preprocessor.process_assets.process_video') # Mock the feature extraction
    @patch('video_preprocessor.utils.get_file_hash') # Mock file hashing
    @patch('video_preprocessor.cv2.VideoCapture') # Mock OpenCV
    def test_preprocess_target_video_new_video(self, mock_videocapture, mock_get_file_hash, 
                                               mock_process_video, mock_db_module):
        if preprocess_target_video is None:
            self.skipTest("video_preprocessor module or function not available.")

        # --- Configure Mocks ---
        # DB Mocks
        mock_db_session_instance = MagicMock()
        mock_db_module._get_db_session.return_value = mock_db_session_instance # If using _get_db_session
        # Or if video_preprocessor directly calls SessionLocal()...
        # mock_db_module.DatabaseSession.return_value = mock_db_session_instance
        
        mock_db_module.is_video_exist.return_value = False # Simulate video not processed
        mock_db_module.add_video.return_value = None # Assume it's a void function
        mock_db_module.get_just_frame_times_by_path.return_value = [0, 1, 2] # Mock frame times in seconds

        # process_assets.process_video mock (feature generator)
        # Yields (frame_time_seconds: float, feature_vector: np.ndarray)
        # The _prepare_features_generator in video_preprocessor will convert these.
        mock_process_video.return_value = iter([
            (0.0, MagicMock(spec=bytes)), # Mock np.ndarray that can be .tobytes()
            (1.0, MagicMock(spec=bytes)),
            (2.0, MagicMock(spec=bytes)),
        ])
        # Ensure the mock ndarray has a .tobytes() method
        for _, mock_ndarray in mock_process_video.return_value:
             # if it's already a MagicMock, configure its tobytes
            if isinstance(mock_ndarray, MagicMock):
                 mock_ndarray.tobytes.return_value = b"dummy_feature_bytes"


        # utils.get_file_hash mock
        mock_get_file_hash.return_value = "dummy_checksum_123"

        # cv2.VideoCapture mock
        mock_cv_cap_instance = MagicMock()
        mock_videocapture.return_value = mock_cv_cap_instance
        mock_cv_cap_instance.isOpened.return_value = True
        mock_cv_cap_instance.get.return_value = 30.0 # Mock FPS
        
        # --- Call the function ---
        result_path_json, result_path_db = preprocess_target_video(self.test_video_path, force_reprocess=False)

        # --- Assertions ---
        mock_db_module.is_video_exist.assert_called_once_with(mock_db_session_instance, self.test_video_path)
        mock_get_file_hash.assert_called_once_with(self.test_video_path) # Assuming ENABLE_CHECKSUM is true
        mock_process_video.assert_called_once_with(self.test_video_path)
        
        # Check that add_video was called. The arguments are a bit complex due to the generator.
        # We can check it was called, and potentially the path, checksum.
        self.assertTrue(mock_db_module.add_video.called)
        args_call_add_video, _ = mock_db_module.add_video.call_args
        self.assertEqual(args_call_add_video[1], self.test_video_path) # path
        self.assertEqual(args_call_add_video[3], "dummy_checksum_123") # checksum

        mock_videocapture.assert_called_once_with(self.test_video_path)
        mock_cv_cap_instance.get.assert_called_with(cv2.CAP_PROP_FPS) # Assuming cv2 is imported or mocked
        mock_db_module.get_just_frame_times_by_path.assert_called() # Called to get times for JSON

        self.assertIsNotNone(result_path_json)
        self.assertTrue(os.path.exists(result_path_json)) # Check if JSON file was created
        
        # Verify JSON content (optional, but good)
        with open(result_path_json, 'r') as f:
            data = json.load(f)
        self.assertEqual(data['fps'], 30.0)
        self.assertEqual(len(data['frames']), 3)
        self.assertEqual(data['frames'][0]['time_sec'], 0.0)
        self.assertEqual(data['frames'][0]['time_hmsf'], "00:00:00:00") # Assuming utils.seconds_to_hmsf is correct

        # Clean up the created JSON by the test itself
        if os.path.exists(result_path_json):
            os.remove(result_path_json)

    # Add more tests:
    # - test_preprocess_target_video_already_processed_no_force()
    # - test_preprocess_target_video_force_reprocess()
    # - test_preprocess_target_video_file_not_found()
    # - test_preprocess_target_video_opencv_error()
    # - test_preprocess_target_video_db_error_on_add()


if __name__ == '__main__':
    # This allows running tests directly from this file if needed
    # but usually, you'd run `python -m unittest discover tests`
    unittest.main()
