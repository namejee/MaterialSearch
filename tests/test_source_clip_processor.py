import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from source_clip_processor import extract_source_clip_frames_and_features
    from capcut_parser import ClipInfo # Needed for creating test ClipInfo objects
except ImportError:
    extract_source_clip_frames_and_features = None
    # Define a dummy ClipInfo if capcut_parser is not available for some reason in test setup
    if 'ClipInfo' not in globals():
        from dataclasses import dataclass
        @dataclass
        class ClipInfo:
            editor_id: str
            source_video_path: str
            source_start_us: int
            source_duration_us: int
            timeline_start_us: int
            target_video_path: str | None = None


# Mock cv2 if not available or for controlled testing
try:
    import cv2
except ImportError:
    cv2 = MagicMock()


class TestSourceClipProcessor(unittest.TestCase):

    def setUp(self):
        self.dummy_video_path = "dummy_source_clip_test_video.mp4"
        # Create a minimal dummy video file.
        # Actual content doesn't matter much if cv2.VideoCapture and process_assets are mocked.
        with open(self.dummy_video_path, "w") as f:
            f.write("dummy video data")

        self.sample_clip_info = ClipInfo(
            editor_id="test_clip_1",
            source_video_path=self.dummy_video_path,
            source_start_us=0, # Start at 0 seconds
            source_duration_us=5_000_000, # 5 seconds duration
            timeline_start_us=0,
            target_video_path="target.mp4"
        )
        self.sample_interval_sec = 1.0 # Sample every 1 second

    def tearDown(self):
        if os.path.exists(self.dummy_video_path):
            os.remove(self.dummy_video_path)

    @patch('source_clip_processor.cv2.VideoCapture')
    @patch('source_clip_processor.process_assets.get_image_feature') # Mock the actual feature extraction
    @patch('source_clip_processor.os.path.exists') # Mock os.path.exists
    def test_extract_frames_and_features_normal_case(self, mock_path_exists, mock_get_image_feature, mock_videocapture):
        if extract_source_clip_frames_and_features is None:
            self.skipTest("source_clip_processor module or function not available.")

        mock_path_exists.return_value = True # Assume video file exists

        # --- Configure Mocks ---
        # cv2.VideoCapture mock
        mock_cv_cap_instance = MagicMock()
        mock_videocapture.return_value = mock_cv_cap_instance
        mock_cv_cap_instance.isOpened.return_value = True
        mock_cv_cap_instance.get.return_value = 30.0 # Mock FPS = 30
        
        # Simulate video.read() returning a frame successfully multiple times, then failing
        # (frame_data, frame_data, ..., None)
        # 5s duration, 1s interval -> 0s, 1s, 2s, 3s, 4s (5 frames)
        mock_frames = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(5) 
        ] + [(False, None)] # Ensure it stops
        mock_cv_cap_instance.read.side_effect = mock_frames
        
        # process_assets.get_image_feature mock
        # Expects a list of frames, returns a list of feature vectors or None
        def mock_feature_extraction_side_effect(frames_list):
            if frames_list and len(frames_list) > 0:
                # Return a list of dummy feature vectors
                return [np.random.rand(512).astype(np.float32) for _ in frames_list]
            return None
        mock_get_image_feature.side_effect = mock_feature_extraction_side_effect

        # --- Call the function ---
        results = extract_source_clip_frames_and_features(
            self.sample_clip_info, 
            self.sample_interval_sec
        )

        # --- Assertions ---
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 5) # Expected 5 frames (0s, 1s, 2s, 3s, 4s)

        # Check timestamps
        expected_timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        for i, (ts, feature) in enumerate(results):
            self.assertAlmostEqual(ts, expected_timestamps[i])
            self.assertIsInstance(feature, np.ndarray)
            self.assertEqual(feature.shape, (512,))

        mock_videocapture.assert_called_once_with(self.dummy_video_path)
        self.assertEqual(mock_cv_cap_instance.set.call_count, 5) # Called for each frame seek
        self.assertEqual(mock_cv_cap_instance.read.call_count, 5) # Called until first failure or end
        self.assertEqual(mock_get_image_feature.call_count, 5) # Called for each successfully read frame

    # Add more tests:
    # - test_video_file_not_found (mock os.path.exists to return False)
    # - test_video_cannot_be_opened (mock cap.isOpened() to return False)
    # - test_invalid_fps (mock cap.get(cv2.CAP_PROP_FPS) to return 0 or None)
    # - test_clip_duration_shorter_than_sample_interval (should yield 1 frame or 0 if start=0, duration<interval)
    # - test_feature_extraction_returns_none (mock get_image_feature to return None)
    # - test_frame_read_fails_midway

if __name__ == '__main__':
    unittest.main()
