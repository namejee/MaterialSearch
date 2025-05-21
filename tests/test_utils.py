import unittest
from unittest.mock import patch, MagicMock
import os # For patching os.path.exists

# Add project root to sys.path to allow direct import of modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import seconds_to_hmsf, get_video_fps, get_file_hash # Assuming get_file_hash is also in utils
# cv2 might not be available in all test environments, so conditional import or full mock
try:
    import cv2
except ImportError:
    cv2 = MagicMock() # Mock cv2 if not installed


class TestUtils(unittest.TestCase):

    def test_seconds_to_hmsf(self):
        self.assertEqual(seconds_to_hmsf(0.0, 30.0), "00:00:00:00")
        self.assertEqual(seconds_to_hmsf(1.0, 30.0), "00:00:01:00")
        self.assertEqual(seconds_to_hmsf(1.0 / 30.0, 30.0), "00:00:00:01") # 1 frame
        self.assertEqual(seconds_to_hmsf(1.5, 30.0), "00:00:01:15") # 1 second and 15 frames
        self.assertEqual(seconds_to_hmsf(3600.0, 30.0), "01:00:00:00")
        self.assertEqual(seconds_to_hmsf(3661.1, 30.0), "01:01:01:03") # 0.1s * 30fps = 3 frames
        
        # Test with 29.97 FPS
        self.assertEqual(seconds_to_hmsf(1.0, 29.97), "00:00:01:00")
        # 1.0333666s for 1 frame at 29.97 (1/29.97 ~ 0.0333666)
        # (1.0333666 - 1.0) * 29.97 = 0.0333666 * 29.97 ~ 1.0 frame
        self.assertEqual(seconds_to_hmsf(1.0333666, 29.97), "00:00:01:01")
        self.assertEqual(seconds_to_hmsf(1.5, 29.97), "00:00:01:15") # 0.5 * 29.97 = 14.985 -> rounded to 15
        
        # Test with 25 FPS
        self.assertEqual(seconds_to_hmsf(1.0, 25.0), "00:00:01:00")
        self.assertEqual(seconds_to_hmsf(1.04, 25.0), "00:00:01:01") # 0.04 * 25 = 1 frame
        
        # Test with 60 FPS
        self.assertEqual(seconds_to_hmsf(0.5, 60.0), "00:00:00:30")
        
        # Test rounding for frames close to next second
        self.assertEqual(seconds_to_hmsf(0.99, 25.0), "00:00:00:25") # 0.99 * 25 = 24.75 -> rounded to 25. Should be 24.
                                                                    # The implementation uses int(round(frac_sec * fps))
                                                                    # (0.99 - 0) * 25 = 24.75 -> round(24.75) = 25.
                                                                    # If fps is 25, frames are 0-24. So 25 should be 00:00:01:00 or 00:00:00:24
                                                                    # Correct behavior for HH:MM:SS:FF usually caps FF at FPS-1
        # Let's re-verify the implementation detail: `if frame_within_second >= fps: frame_within_second = int(fps-1)`
        # So, if 0.99 * 25 = 24.75 -> round = 25. Then 25 >= 25, so frame_within_second = 24.
        self.assertEqual(seconds_to_hmsf(0.99, 25.0), "00:00:00:24") # Corrected based on implementation detail.

        # Test invalid FPS (should default to 25 FPS with a warning, which we can't easily check here)
        self.assertEqual(seconds_to_hmsf(1.0, 0), "00:00:01:00") # Defaults to 25 fps
        self.assertEqual(seconds_to_hmsf(1.04, 0), "00:00:01:01") # 0.04 * 25 = 1

    @patch('utils.cv2.VideoCapture') # Patching cv2.VideoCapture within the utils module
    @patch('utils.os.path.exists')    # Patching os.path.exists within the utils module
    def test_get_video_fps(self, mock_exists, mock_videocapture):
        mock_exists.return_value = True # Assume file exists for most tests

        # Mock VideoCapture object
        mock_cap_instance = MagicMock()
        mock_videocapture.return_value = mock_cap_instance

        # Test case 1: Valid FPS
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.return_value = 29.97
        self.assertEqual(get_video_fps("dummy.mp4"), 29.97)
        mock_cap_instance.get.assert_called_once_with(cv2.CAP_PROP_FPS)
        mock_cap_instance.release.assert_called_once()

        # Reset mocks for next test case
        mock_cap_instance.reset_mock()
        mock_videocapture.reset_mock() # Reset the constructor mock as well
        mock_videocapture.return_value = mock_cap_instance # Re-assign for subsequent calls

        # Test case 2: FPS is 0
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.return_value = 0.0
        self.assertIsNone(get_video_fps("dummy.mp4"))
        mock_cap_instance.release.assert_called_once()
        
        mock_cap_instance.reset_mock()
        mock_videocapture.reset_mock()
        mock_videocapture.return_value = mock_cap_instance

        # Test case 3: FPS is None (some backends might return None)
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.return_value = None
        self.assertIsNone(get_video_fps("dummy.mp4"))
        mock_cap_instance.release.assert_called_once()

        mock_cap_instance.reset_mock()
        mock_videocapture.reset_mock()
        mock_videocapture.return_value = mock_cap_instance

        # Test case 4: Video file cannot be opened
        mock_cap_instance.isOpened.return_value = False
        self.assertIsNone(get_video_fps("dummy.mp4"))
        # release() might not be called if isOpened() is false, depends on implementation.
        # Current utils.py implementation calls release in finally block if cap was assigned.
        # But if cap = cv2.VideoCapture itself failed, it wouldn't. Let's assume VideoCapture returns an object.
        # The mock_cap_instance is returned by mock_videocapture, so cap.release() will be called.
        mock_cap_instance.release.assert_called_once()


        mock_cap_instance.reset_mock()
        mock_videocapture.reset_mock()
        mock_videocapture.return_value = mock_cap_instance
        
        # Test case 5: File does not exist
        mock_exists.return_value = False
        self.assertIsNone(get_video_fps("non_existent.mp4"))
        mock_videocapture.assert_not_called() # VideoCapture should not be called if file doesn't exist

        # Test case 6: Exception during cv2.VideoCapture or methods
        mock_exists.return_value = True
        mock_videocapture.side_effect = Exception("CV2 Read Error")
        self.assertIsNone(get_video_fps("dummy_error.mp4"))

    def test_get_file_hash(self):
        # This test involves actual file I/O, which is okay for utils like this.
        # Create a dummy file
        dummy_file_content = "This is a test file for hashing."
        dummy_file_path = "test_hash_file.txt"
        with open(dummy_file_path, "w") as f:
            f.write(dummy_file_content)

        # Calculate hash (SHA1 as per existing utils.py)
        # Expected SHA1 for "This is a test file for hashing."
        expected_hash = "a8a82ce3c1c7d4e80fac1299993ff76753117803" 
        
        actual_hash = get_file_hash(dummy_file_path)
        self.assertEqual(actual_hash, expected_hash)

        # Test non-existent file
        os.remove(dummy_file_path) # Clean up before next assertion
        # The current get_file_hash in utils.py logs an error and returns None
        self.assertIsNone(get_file_hash("non_existent_hash_file.txt"))


if __name__ == '__main__':
    unittest.main()
