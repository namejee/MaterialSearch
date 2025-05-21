import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from output_formatter import format_match_results, print_formatted_results, save_results_to_json
from clip_matcher import MatchResult # Assuming these are importable
from capcut_parser import CapCutProject, ClipInfo # Assuming these are importable
# utils itself is not directly tested here, but its functions are mocked via output_formatter's imports

class TestOutputFormatter(unittest.TestCase):

    def setUp(self):
        self.mock_capcut_project = CapCutProject(
            source_clips=[
                ClipInfo(editor_id="clip1", source_video_path="/path/src1.mp4", 
                         source_start_us=1000000, source_duration_us=5000000, timeline_start_us=0),
                ClipInfo(editor_id="clip2", source_video_path="/path/src2.mp4", 
                         source_start_us=2000000, source_duration_us=3000000, timeline_start_us=6000000), # 6s timeline start
                ClipInfo(editor_id="clip3", source_video_path="/path/src3.mp4", 
                         source_start_us=0, source_duration_us=4000000, timeline_start_us=2000000), # 2s timeline start
            ]
        )
        self.mock_match_results = [
            MatchResult( # Belongs to clip2, should appear third after sorting
                source_clip_editor_id="clip2", source_video_path="/path/src2.mp4", source_timeline_start_us=6000000,
                source_clip_start_sec=2.0, source_clip_end_sec=4.5, 
                target_video_start_sec=20.0, target_video_end_sec=22.5,
                average_similarity=0.92345, num_matched_samples=10
            ),
            MatchResult( # Belongs to clip1, should appear first
                source_clip_editor_id="clip1", source_video_path="/path/src1.mp4", source_timeline_start_us=0,
                source_clip_start_sec=1.0, source_clip_end_sec=5.0,
                target_video_start_sec=10.0, target_video_end_sec=14.0,
                average_similarity=0.8876, num_matched_samples=15
            ),
             MatchResult( # Belongs to clip3, should appear second
                source_clip_editor_id="clip3", source_video_path="/path/src3.mp4", source_timeline_start_us=2000000,
                source_clip_start_sec=0.5, source_clip_end_sec=3.5, 
                target_video_start_sec=5.0, target_video_end_sec=8.0,
                average_similarity=0.9511, num_matched_samples=12
            ),
        ]
        self.target_video_fps = 29.97

    @patch('output_formatter.get_video_fps') # Patch get_video_fps where it's used in output_formatter
    @patch('output_formatter.seconds_to_hmsf') # Patch seconds_to_hmsf for predictable output
    def test_format_match_results_structure_and_sorting(self, mock_seconds_to_hmsf, mock_get_video_fps):
        # Configure mocks
        def mock_hmsf_side_effect(seconds, fps):
            return f"{seconds:.2f}s_at_{fps:.2f}fps_hmsf" # Predictable mock output
        mock_seconds_to_hmsf.side_effect = mock_hmsf_side_effect
        
        # Mock FPS for different source videos
        def mock_fps_side_effect(video_path):
            if video_path == "/path/src1.mp4": return 30.0
            if video_path == "/path/src2.mp4": return 25.0
            if video_path == "/path/src3.mp4": return 60.0
            return None
        mock_get_video_fps.side_effect = mock_fps_side_effect

        formatted_results = format_match_results(
            self.mock_match_results, 
            self.mock_capcut_project, 
            self.target_video_fps
        )

        self.assertEqual(len(formatted_results), 3)

        # --- Check Sorting (based on original_clip_info.timeline_start_us) ---
        # Expected order: clip1 (0us), clip3 (2000000us), clip2 (6000000us)
        self.assertEqual(formatted_results[0]["编辑器ID"], "clip1")
        self.assertEqual(formatted_results[1]["编辑器ID"], "clip3")
        self.assertEqual(formatted_results[2]["编辑器ID"], "clip2")

        # --- Check Structure and Content of the first result (clip1) ---
        res1 = formatted_results[0]
        self.assertEqual(res1["编辑器ID"], "clip1")
        # timeline_start_us = 0 for clip1, source_fps = 30.0
        self.assertEqual(res1["剪映时间轴起始时间"], "0.00s_at_30.00fps_hmsf") 
        
        # original_clip_info.source_start_us = 1000000 (1s), duration = 5000000 (5s) -> end = 6s
        # source_fps = 30.0
        expected_orig_src_start_hmsf = "1.00s_at_30.00fps_hmsf"
        expected_orig_src_end_hmsf = "6.00s_at_30.00fps_hmsf" # 1s + 5s = 6s
        self.assertEqual(res1["剪辑源视频原始起止时间"], f"{expected_orig_src_start_hmsf} - {expected_orig_src_end_hmsf}")

        # target_video_start_sec=10.0, target_video_end_sec=14.0, target_fps = 29.97
        expected_matched_target_start_hmsf = "10.00s_at_29.97fps_hmsf"
        expected_matched_target_end_hmsf = "14.00s_at_29.97fps_hmsf"
        self.assertEqual(res1["匹配的原片视频起止时间"], f"{expected_matched_target_start_hmsf} - {expected_matched_target_end_hmsf}")
        
        self.assertEqual(res1["平均相似度"], "88.76%") # 0.8876
        self.assertEqual(res1["采样点数量"], 15)

        # --- Check content of the second result (clip3) ---
        res2 = formatted_results[1]
        self.assertEqual(res2["编辑器ID"], "clip3")
        # timeline_start_us = 2000000 (2s) for clip3, source_fps = 60.0
        self.assertEqual(res2["剪映时间轴起始时间"], "2.00s_at_60.00fps_hmsf")

        # original_clip_info.source_start_us = 0 (0s), duration = 4000000 (4s) -> end = 4s
        # source_fps = 60.0
        expected_orig_src_start_hmsf_c3 = "0.00s_at_60.00fps_hmsf"
        expected_orig_src_end_hmsf_c3 = "4.00s_at_60.00fps_hmsf" # 0s + 4s = 4s
        self.assertEqual(res2["剪辑源视频原始起止时间"], f"{expected_orig_src_start_hmsf_c3} - {expected_orig_src_end_hmsf_c3}")

        # target_video_start_sec=5.0, target_video_end_sec=8.0, target_fps = 29.97
        expected_matched_target_start_hmsf_c3 = "5.00s_at_29.97fps_hmsf"
        expected_matched_target_end_hmsf_c3 = "8.00s_at_29.97fps_hmsf"
        self.assertEqual(res2["匹配的原片视频起止时间"], f"{expected_matched_target_start_hmsf_c3} - {expected_matched_target_end_hmsf_c3}")

        self.assertEqual(res2["平均相似度"], "95.11%")
        self.assertEqual(res2["采样点数量"], 12)


    @patch('output_formatter.get_video_fps')
    def test_format_match_results_missing_clipinfo_or_fps(self, mock_get_video_fps):
        # Case 1: ClipInfo missing for a MatchResult
        match_results_missing_clip = [
            MatchResult(source_clip_editor_id="unknown_clip", source_video_path="/path/src_unknown.mp4", 
                        source_timeline_start_us=0, source_clip_start_sec=1.0, source_clip_end_sec=2.0,
                        target_video_start_sec=1.0,target_video_end_sec=2.0, average_similarity=0.9, num_matched_samples=5)
        ]
        formatted = format_match_results(match_results_missing_clip, self.mock_capcut_project, self.target_video_fps)
        self.assertEqual(len(formatted), 0) # Should skip the result

        # Case 2: get_video_fps returns None for a source video
        mock_get_video_fps.return_value = None # Simulate failure to get FPS
        
        # Use only clip1 for this test, which exists in mock_capcut_project
        single_match_result = [self.mock_match_results[1]] # This is clip1
        
        with patch('output_formatter.seconds_to_hmsf') as mock_hmsf:
            def hmsf_default_fps_check(seconds, fps):
                # Check if the fallback FPS (30.0) is used for source timings
                return f"{seconds:.2f}s_at_{fps:.2f}fps_default_check"
            mock_hmsf.side_effect = hmsf_default_fps_check

            formatted_default_fps = format_match_results(single_match_result, self.mock_capcut_project, self.target_video_fps)
            self.assertEqual(len(formatted_default_fps), 1)
            res = formatted_default_fps[0]
            
            # Check if 30.0 (default fallback) was used for source/timeline times
            # timeline_start_us = 0 for clip1
            self.assertIn("0.00s_at_30.00fps_default_check", res["剪映时间轴起始时间"])
            # original_clip_info.source_start_us = 1000000 (1s) for clip1
            self.assertIn("1.00s_at_30.00fps_default_check", res["剪辑源视频原始起止时间"])
            # Check that target FPS is still used for target times
            self.assertIn(f"at_{self.target_video_fps:.2f}fps_default_check", res["匹配的原片视频起止时间"])
            
            # Ensure get_video_fps was called for clip1's source video path
            mock_get_video_fps.assert_called_with("/path/src1.mp4")


    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.dump')
    def test_save_results_to_json(self, mock_json_dump, mock_file_open):
        dummy_results = [{"id": "test", "data": "some_data"}]
        output_path = "test_output.json"
        
        save_results_to_json(dummy_results, output_path)
        
        mock_file_open.assert_called_once_with(output_path, 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(dummy_results, mock_file_open(), ensure_ascii=False, indent=4)

    # Test for print_formatted_results can be done by patching builtins.print
    # but it's often simpler to visually inspect or test the string formatting logic indirectly
    # via format_match_results as done above.

if __name__ == '__main__':
    unittest.main()
