import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clip_matcher import find_matching_segments, MatchResult, _cosine_similarity
# Assuming database.py and models.py are structured such that these imports work
# or that the mock setup within clip_matcher.py itself handles their absence for testing.

class TestClipMatcher(unittest.TestCase):

    def _create_mock_features(self, num_features, dim=3):
        # Creates simple, somewhat distinct features for testing
        features = []
        for i in range(num_features):
            arr = np.zeros(dim, dtype=np.float32)
            if i < dim:
                arr[i] = 1.0
            else:
                arr[i % dim] = 0.5 # Make it different
                arr[(i+1) % dim] = 0.5
            # Normalize (though simple 0/1 vectors are already somewhat normalized)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr /= norm
            features.append(arr)
        return features

    def test_cosine_similarity(self):
        # Test helper directly as it's critical
        feat1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        feat2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        feat3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        feat4 = np.array([0.707, 0.707, 0.0], dtype=np.float32) # Normalized approx of [1,1,0]

        self.assertAlmostEqual(_cosine_similarity(feat1, feat2), 1.0)
        self.assertAlmostEqual(_cosine_similarity(feat1, feat3), 0.0)
        self.assertAlmostEqual(_cosine_similarity(feat1, feat4), 0.707, places=3)
        
        # Test with non-normalized (though function assumes normalized)
        feat5 = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        # _cosine_similarity will produce np.dot(feat1, feat5.T) which is 2.0.
        # If they were normalized, it would be 1.0. The current helper does not normalize internally.
        # This confirms the assumption that features should be pre-normalized.
        self.assertAlmostEqual(np.dot(feat1, feat5.T), 2.0)


    @patch('clip_matcher.get_db') # Mocks get_db context manager used in find_matching_segments
    def test_find_matching_segments_perfect_match(self, mock_get_db_context):
        # --- Setup Mock DB ---
        mock_db_session = MagicMock()
        # Configure the context manager __enter__ to return the mock session
        mock_get_db_context.return_value.__enter__.return_value = mock_db_session
        
        # Mock return value for get_frame_times_features_by_path
        target_features_np = self._create_mock_features(5) # T0, T1, T2, T3, T4
        target_timestamps_int = [0, 1, 2, 3, 4] # Target video frames at 0s, 1s, 2s, etc.
        target_features_bytes = [f.tobytes() for f in target_features_np]
        
        # This is the function that will be called inside find_matching_segments
        # We need to ensure it's patched correctly, or that the mock_db_session handles it.
        # The find_matching_segments calls "get_frame_times_features_by_path(db_session, target_video_path)"
        # So, if we import it as "from database import get_frame_times_features_by_path"
        # then we need to patch 'clip_matcher.get_frame_times_features_by_path'
        
        with patch('clip_matcher.get_frame_times_features_by_path') as mock_db_call:
            mock_db_call.return_value = (target_timestamps_int, target_features_bytes)

            # --- Source Features ---
            # Source clip matches T0, T1, T2 perfectly
            source_frames = [
                (10.0, target_features_np[0]), # S0 matches T0
                (11.0, target_features_np[1]), # S1 matches T1
                (12.0, target_features_np[2]), # S2 matches T2
            ]

            results = find_matching_segments(
                source_frames_with_features=source_frames,
                target_video_path="dummy_target.mp4",
                similarity_threshold=0.95,
                min_match_sequence_len=3,
                max_time_drift_sec=0.1,
                source_clip_editor_id="clip1",
                source_video_path_from_clip="/src.mp4",
                source_timeline_start_us_from_clip=0
            )
            self.assertEqual(len(results), 1)
            match = results[0]
            self.assertEqual(match.num_matched_samples, 3)
            self.assertAlmostEqual(match.average_similarity, 1.0, places=5)
            self.assertEqual(match.source_frame_timestamps, [10.0, 11.0, 12.0])
            self.assertEqual(match.target_frame_timestamps, [0.0, 1.0, 2.0]) # Timestamps are float after conversion
            mock_db_call.assert_called_once_with(mock_db_session, "dummy_target.mp4")


    @patch('clip_matcher.get_db')
    def test_no_match_found(self, mock_get_db_context):
        mock_db_session = MagicMock()
        mock_get_db_context.return_value.__enter__.return_value = mock_db_session
        
        target_features_np = self._create_mock_features(5, dim=3) # T0-T4
        target_timestamps_int = [0,1,2,3,4]
        target_features_bytes = [f.tobytes() for f in target_features_np]

        with patch('clip_matcher.get_frame_times_features_by_path') as mock_db_call:
            mock_db_call.return_value = (target_timestamps_int, target_features_bytes)

            source_features_np = self._create_mock_features(3, dim=4) # Different dimension, will not match
            source_frames = [
                (0.0, source_features_np[0]), (0.5, source_features_np[1]), (1.0, source_features_np[2])
            ]
            # Or features that are simply very different
            source_frames_diff = [
                (0.0, np.array([-1.0, 0.1, 0.1], dtype=np.float32)),
                (0.5, np.array([-0.9, -0.1, 0.2], dtype=np.float32)),
                (1.0, np.array([-0.8, 0.2, -0.1], dtype=np.float32)),
            ]


            results = find_matching_segments(
                source_frames_with_features=source_frames_diff,
                target_video_path="dummy_target.mp4",
                similarity_threshold=0.95, min_match_sequence_len=2, max_time_drift_sec=0.1,
                source_clip_editor_id="c1", source_video_path_from_clip="/s.mp4", source_timeline_start_us_from_clip=0
            )
            self.assertEqual(len(results), 0)

    @patch('clip_matcher.get_db')
    def test_match_shorter_than_min_len(self, mock_get_db_context):
        mock_db_session = MagicMock()
        mock_get_db_context.return_value.__enter__.return_value = mock_db_session
        
        target_features_np = self._create_mock_features(5)
        target_timestamps_int = [i for i in range(5)]
        target_features_bytes = [f.tobytes() for f in target_features_np]

        with patch('clip_matcher.get_frame_times_features_by_path') as mock_db_call:
            mock_db_call.return_value = (target_timestamps_int, target_features_bytes)
            
            source_frames = [ # Only 2 frames match
                (0.0, target_features_np[0]), (0.5, target_features_np[1]) 
            ]
            results = find_matching_segments(
                source_frames_with_features=source_frames,
                target_video_path="dummy_target.mp4",
                similarity_threshold=0.95, min_match_sequence_len=3, max_time_drift_sec=0.1, # min_len is 3
                source_clip_editor_id="c1", source_video_path_from_clip="/s.mp4", source_timeline_start_us_from_clip=0
            )
            self.assertEqual(len(results), 0)

    @patch('clip_matcher.get_db')
    def test_match_below_similarity_threshold(self, mock_get_db_context):
        mock_db_session = MagicMock()
        mock_get_db_context.return_value.__enter__.return_value = mock_db_session

        target_features_np = self._create_mock_features(3) # T0, T1, T2
        target_timestamps_int = [0,1,2]
        target_features_bytes = [f.tobytes() for f in target_features_np]

        with patch('clip_matcher.get_frame_times_features_by_path') as mock_db_call:
            mock_db_call.return_value = (target_timestamps_int, target_features_bytes)

            # Create source features that are somewhat similar but not enough
            s_feat0 = target_features_np[0] * 0.5 + self._create_mock_features(1, dim=3)[0] * 0.5
            s_feat1 = target_features_np[1] * 0.5 + self._create_mock_features(1, dim=3)[0] * 0.5 # Different from T1
            s_feat2 = target_features_np[2] * 0.5 + self._create_mock_features(1, dim=3)[0] * 0.5
            # Normalize them to ensure dot product is cosine similarity
            s_feat0 /= np.linalg.norm(s_feat0)
            s_feat1 /= np.linalg.norm(s_feat1)
            s_feat2 /= np.linalg.norm(s_feat2)
            
            source_frames = [
                (0.0, s_feat0), (0.5, s_feat1), (1.0, s_feat2)
            ]
            # Similarity will be around 0.5-0.7 if features are mixed this way (depending on mock_features)
            # Let's assume _cosine_similarity(s_feat0, target_features_np[0]) is ~0.7
            # print("Sim for threshold test:", _cosine_similarity(s_feat0, target_features_np[0]))

            results = find_matching_segments(
                source_frames_with_features=source_frames,
                target_video_path="dummy_target.mp4",
                similarity_threshold=0.90, # Set a high threshold
                min_match_sequence_len=2, max_time_drift_sec=0.1,
                source_clip_editor_id="c1", source_video_path_from_clip="/s.mp4", source_timeline_start_us_from_clip=0
            )
            self.assertEqual(len(results), 0)

    @patch('clip_matcher.get_db')
    def test_time_drift_violation(self, mock_get_db_context):
        mock_db_session = MagicMock()
        mock_get_db_context.return_value.__enter__.return_value = mock_db_session

        target_features_np = self._create_mock_features(5)
        target_timestamps_int = [0, 1, 3, 4, 5] # T2 is at 3s, creating a 2s gap from T1 (at 1s)
        target_features_bytes = [f.tobytes() for f in target_features_np]

        with patch('clip_matcher.get_frame_times_features_by_path') as mock_db_call:
            mock_db_call.return_value = (target_timestamps_int, target_features_bytes)
        
            source_frames = [ # S0, S1, S2 should match T0, T1, T2 in features
                (10.0, target_features_np[0]), # S0 -> T0 (ts=0)
                (10.5, target_features_np[1]), # S1 -> T1 (ts=1). Source diff=0.5s, Target diff=1s. Drift=0.5s
                (11.0, target_features_np[2]), # S2 -> T2 (ts=3). Source diff=0.5s, Target diff=2s. Drift=1.5s
            ]

            # Case 1: max_time_drift_sec = 0.6. S0-T0, S1-T1 should match. S2-T2 should fail drift.
            results = find_matching_segments(
                source_frames_with_features=source_frames,
                target_video_path="dummy_target.mp4",
                similarity_threshold=0.95, min_match_sequence_len=2, max_time_drift_sec=0.6,
                source_clip_editor_id="c1", source_video_path_from_clip="/s.mp4", source_timeline_start_us_from_clip=0
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].num_matched_samples, 2) # S0-T0, S1-T1
            self.assertEqual(results[0].source_frame_timestamps, [10.0, 10.5])
            self.assertEqual(results[0].target_frame_timestamps, [0.0, 1.0])

            # Case 2: max_time_drift_sec = 0.1. S0-T0 is initial. S1-T1 fails drift (0.5s > 0.1s)
            results_strict_drift = find_matching_segments(
                source_frames_with_features=source_frames,
                target_video_path="dummy_target.mp4",
                similarity_threshold=0.95, min_match_sequence_len=2, max_time_drift_sec=0.1,
                source_clip_editor_id="c1", source_video_path_from_clip="/s.mp4", source_timeline_start_us_from_clip=0
            )
            # No sequence of length 2 can be formed.
            # S0,T0 match. S1,T1: S_diff=0.5, T_diff=1.0. Drift=0.5. Fails.
            # Sequence S0,T0 is length 1, not enough.
            # Then it tries S1,T_any. S1,T1 match. S2,T2: S_diff=0.5, T_diff=2.0. Drift=1.5. Fails.
            # Sequence S1,T1 is length 1, not enough.
            self.assertEqual(len(results_strict_drift), 0)


    @patch('clip_matcher.get_db')
    def test_multiple_matches_and_greedy_behavior(self, mock_get_db_context):
        mock_db_session = MagicMock()
        mock_get_db_context.return_value.__enter__.return_value = mock_db_session

        # T0,T1,T2 form one sequence. T3,T4 form another.
        target_features_np = self._create_mock_features(5) 
        target_timestamps_int = [0,1,2,  10,11] # Target features for two segments
        target_features_bytes = [f.tobytes() for f in target_features_np]

        with patch('clip_matcher.get_frame_times_features_by_path') as mock_db_call:
            mock_db_call.return_value = (target_timestamps_int, target_features_bytes)

            # Source has two segments matching the two target segments
            source_frames = [
                (0.0, target_features_np[0]), # S0 -> T0
                (0.5, target_features_np[1]), # S1 -> T1
                (1.0, target_features_np[2]), # S2 -> T2
                (5.0, np.array([0.1,0.1,0.1])), # S3 - non-matching separator
                (10.0, target_features_np[3]),# S4 -> T3 (at 10s)
                (10.5, target_features_np[4]),# S5 -> T4 (at 11s)
            ]
            results = find_matching_segments(
                source_frames_with_features=source_frames,
                target_video_path="dummy_target.mp4",
                similarity_threshold=0.95, min_match_sequence_len=2, max_time_drift_sec=0.6,
                source_clip_editor_id="c1", source_video_path_from_clip="/s.mp4", source_timeline_start_us_from_clip=0
            )
            self.assertEqual(len(results), 2)
            # Match 1: S0,S1,S2 with T0,T1,T2
            self.assertEqual(results[0].num_matched_samples, 3)
            self.assertEqual(results[0].source_frame_timestamps, [0.0, 0.5, 1.0])
            self.assertEqual(results[0].target_frame_timestamps, [0.0, 1.0, 2.0])
            # Match 2: S4,S5 with T3,T4
            self.assertEqual(results[1].num_matched_samples, 2)
            self.assertEqual(results[1].source_frame_timestamps, [10.0, 10.5])
            self.assertEqual(results[1].target_frame_timestamps, [10.0, 11.0])

    @patch('clip_matcher.get_db')
    def test_empty_source_or_target_frames(self, mock_get_db_context):
        mock_db_session = MagicMock()
        mock_get_db_context.return_value.__enter__.return_value = mock_db_session

        target_features_np = self._create_mock_features(3)
        target_timestamps_int = [0,1,2]
        target_features_bytes = [f.tobytes() for f in target_features_np]

        with patch('clip_matcher.get_frame_times_features_by_path') as mock_db_call:
            # Case 1: Empty source_frames
            mock_db_call.return_value = (target_timestamps_int, target_features_bytes)
            results_empty_source = find_matching_segments(
                source_frames_with_features=[], # Empty source
                target_video_path="dummy.mp4", similarity_threshold=0.9, min_match_sequence_len=2, max_time_drift_sec=0.1,
                source_clip_editor_id="c1", source_video_path_from_clip="/s.mp4", source_timeline_start_us_from_clip=0
            )
            self.assertEqual(len(results_empty_source), 0)

            # Case 2: Empty target_frames from DB
            mock_db_call.return_value = ([], []) # DB returns no frames
            source_frames = [(0.0, target_features_np[0])] # Non-empty source
            results_empty_target = find_matching_segments(
                source_frames_with_features=source_frames,
                target_video_path="dummy_empty_target.mp4", similarity_threshold=0.9, min_match_sequence_len=1, max_time_drift_sec=0.1,
                source_clip_editor_id="c1", source_video_path_from_clip="/s.mp4", source_timeline_start_us_from_clip=0
            )
            self.assertEqual(len(results_empty_target), 0)
            mock_db_call.assert_called_with(mock_db_session, "dummy_empty_target.mp4")


if __name__ == '__main__':
    unittest.main()
