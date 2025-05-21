import json
import logging
from typing import List, Dict, Any

# Assuming these modules are in the python path or same directory structure
try:
    from capcut_parser import CapCutProject, ClipInfo
except ImportError:
    logging.warning("Could not import from capcut_parser. Using placeholders for script structure.")
    from dataclasses import dataclass, field
    @dataclass
    class ClipInfo: # type: ignore
        editor_id: str
        source_video_path: str
        source_start_us: int
        source_duration_us: int
        timeline_start_us: int
        target_video_path: str | None = None
    @dataclass
    class CapCutProject: # type: ignore
        source_clips: List[ClipInfo] = field(default_factory=list)
        # ... other fields

try:
    from clip_matcher import MatchResult
except ImportError:
    logging.warning("Could not import from clip_matcher. Using placeholders for script structure.")
    @dataclass
    class MatchResult: # type: ignore
        source_clip_editor_id: str
        source_video_path: str
        source_timeline_start_us: int 
        source_clip_start_sec: float
        source_clip_end_sec: float
        target_video_start_sec: float
        target_video_end_sec: float
        average_similarity: float
        num_matched_samples: int
        source_frame_timestamps: List[float] = field(default_factory=list)
        target_frame_timestamps: List[float] = field(default_factory=list)

try:
    from utils import seconds_to_hmsf, get_video_fps
except ImportError:
    logging.warning("Could not import from utils. Using placeholders for script structure.")
    def seconds_to_hmsf(seconds: float, fps: float) -> str: # type: ignore
        return f"{int(seconds)}s_mock_hmsf"
    def get_video_fps(video_path: str) -> float | None: # type: ignore
        return 30.0 # Mock FPS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_match_results(
    match_results: List[MatchResult], 
    capcut_project: CapCutProject, 
    target_video_fps: float
) -> List[Dict[str, Any]]:
    """
    Formats the list of MatchResult objects into a human-readable list of dictionaries,
    including timecode conversions.

    Args:
        match_results: A list of MatchResult objects.
        capcut_project: The CapCutProject object containing original ClipInfo.
        target_video_fps: The FPS of the target video.

    Returns:
        A list of dictionaries, each representing a formatted match result,
        sorted by the original timeline_start_us of the clips.
    """
    formatted_outputs_with_sort_key: List[Tuple[int, Dict[str, Any]]] = []

    clip_info_map: Dict[str, ClipInfo] = {
        clip.editor_id: clip for clip in capcut_project.source_clips
    }

    for match_result in match_results:
        original_clip_info = clip_info_map.get(match_result.source_clip_editor_id)

        if not original_clip_info:
            logger.warning(f"Could not find original ClipInfo for editor_id: {match_result.source_clip_editor_id}. Skipping this match.")
            continue

        source_video_fps = get_video_fps(match_result.source_video_path)
        if source_video_fps is None:
            logger.warning(f"Could not get FPS for source video: {match_result.source_video_path}. "
                           "Timecode conversions for source timings will use a default of 30 FPS.")
            source_video_fps = 30.0 # Fallback FPS for HMSF conversion

        # Timeline and original source clip timings (from CapCut project)
        # These use the FPS of the source video as a proxy if project FPS is not known.
        timeline_start_sec = original_clip_info.timeline_start_us / 1_000_000.0
        timeline_start_hmsf = seconds_to_hmsf(timeline_start_sec, source_video_fps)

        source_original_start_sec = original_clip_info.source_start_us / 1_000_000.0
        source_original_end_sec = (original_clip_info.source_start_us + original_clip_info.source_duration_us) / 1_000_000.0
        
        source_clip_original_start_hmsf = seconds_to_hmsf(source_original_start_sec, source_video_fps)
        source_clip_original_end_hmsf = seconds_to_hmsf(source_original_end_sec, source_video_fps)

        # Matched target segment timings (from matching process, relative to target video)
        # These use the target_video_fps.
        # Note: match_result.source_clip_start_sec and source_clip_end_sec are already in seconds
        # and refer to time within the source_video_path. Their HMSF could be generated too if needed.
        # For "匹配的原片视频起止时间", the problem asks for target video times.
        matched_target_start_hmsf = seconds_to_hmsf(match_result.target_video_start_sec, target_video_fps)
        matched_target_end_hmsf = seconds_to_hmsf(match_result.target_video_end_sec, target_video_fps)
        
        formatted_dict = {
            "编辑器ID": match_result.source_clip_editor_id,
            "剪映时间轴起始时间": timeline_start_hmsf,
            "剪辑源视频原始起止时间": f"{source_clip_original_start_hmsf} - {source_clip_original_end_hmsf}",
            "匹配的原片视频起止时间": f"{matched_target_start_hmsf} - {matched_target_end_hmsf}",
            "平均相似度": f"{match_result.average_similarity:.2%}", # Format as percentage
            "采样点数量": match_result.num_matched_samples,
            # Optional debug fields (can be uncommented if needed)
            # "debug_source_video_path": match_result.source_video_path,
            # "debug_source_fps": source_video_fps,
            # "debug_target_fps": target_video_fps,
            # "debug_original_timeline_start_us": original_clip_info.timeline_start_us,
            # "debug_original_source_start_us": original_clip_info.source_start_us,
            # "debug_original_source_duration_us": original_clip_info.source_duration_us,
            # "debug_match_source_start_sec": match_result.source_clip_start_sec, # Matched segment in source
            # "debug_match_source_end_sec": match_result.source_clip_end_sec,   # Matched segment in source
        }
        formatted_outputs_with_sort_key.append((original_clip_info.timeline_start_us, formatted_dict))

    # Sort results based on the original timeline_start_us
    formatted_outputs_with_sort_key.sort(key=lambda x: x[0])

    # Return only the dictionaries after sorting
    return [item[1] for item in formatted_outputs_with_sort_key]


def save_results_to_json(formatted_results: List[Dict[str, Any]], output_file_path: str) -> None:
    """Saves the formatted results to a JSON file."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Formatted results saved to JSON file: {output_file_path}")
    except IOError as e:
        logger.error(f"Failed to write results to JSON file {output_file_path}: {e}", exc_info=True)
    except TypeError as e: # Handle potential non-serializable data if any
        logger.error(f"Data is not JSON serializable for {output_file_path}: {e}", exc_info=True)


def print_formatted_results(formatted_results: List[Dict[str, Any]]) -> None:
    """Prints the formatted results neatly to the console."""
    if not formatted_results:
        print("No match results to display.")
        return

    for i, result in enumerate(formatted_results):
        print(f"\n--- Match {i+1} ---")
        for key, value in result.items():
            print(f"  {key}: {value}")
    print("\n--- End of Results ---")


if __name__ == '__main__':
    logger.info("Running test for output_formatter.py")

    # Create mock CapCutProject
    mock_clips = [
        ClipInfo(editor_id="clip_A", source_video_path="/path/sourceA.mp4", 
                 source_start_us=10_000_000, source_duration_us=5_000_000, timeline_start_us=0), # 10s, 5s duration
        ClipInfo(editor_id="clip_B", source_video_path="/path/sourceB.mp4", 
                 source_start_us=2_000_000, source_duration_us=3_000_000, timeline_start_us=5_000_000), # 2s, 3s duration
    ]
    mock_project = CapCutProject(source_clips=mock_clips)

    # Create mock MatchResult objects
    # These would typically come from clip_matcher.py
    mock_match_results = [
        MatchResult(
            source_clip_editor_id="clip_B", # Earlier in timeline
            source_video_path="/path/sourceB.mp4", # Corresponds to clip_B
            source_timeline_start_us=5_000_000, # Pass-through from original ClipInfo via matcher
            source_clip_start_sec=2.0, # Matched part within sourceB.mp4
            source_clip_end_sec=4.5,   # Matched part within sourceB.mp4
            target_video_start_sec=20.0,
            target_video_end_sec=22.5,
            average_similarity=0.92345,
            num_matched_samples=10
        ),
        MatchResult(
            source_clip_editor_id="clip_A", # Later in timeline
            source_video_path="/path/sourceA.mp4", # Corresponds to clip_A
            source_timeline_start_us=0, # Pass-through
            source_clip_start_sec=10.0, # Matched part within sourceA.mp4
            source_clip_end_sec=14.0,  # Matched part within sourceA.mp4
            target_video_start_sec=30.5,
            target_video_end_sec=34.5,
            average_similarity=0.8876,
            num_matched_samples=15
        ),
    ]

    # Mock get_video_fps - in a real scenario, this would read a file
    # For this test, let's assume sourceA.mp4 is 30fps, sourceB.mp4 is 25fps
    def mock_get_video_fps_test(video_path: str) -> float | None:
        if "sourceA.mp4" in video_path:
            return 30.0
        if "sourceB.mp4" in video_path:
            return 25.0
        return None
    
    # Monkey patch utils.get_video_fps for this test
    # This requires utils to be imported in a way that allows patching, or direct patch if already imported.
    # For simplicity, if 'utils' is the module where get_video_fps is, we'd do:
    # import utils as real_utils
    # real_utils.get_video_fps = mock_get_video_fps_test
    # However, the current structure imports get_video_fps directly.
    # So, we need to patch it in the current module's scope if it was imported.
    _original_get_video_fps = get_video_fps
    globals()['get_video_fps'] = mock_get_video_fps_test
    
    target_fps_for_test = 29.97

    logger.info("Formatting mock match results...")
    formatted = format_match_results(mock_match_results, mock_project, target_fps_for_test)
    
    # Restore original function
    globals()['get_video_fps'] = _original_get_video_fps

    print("\n--- Formatted Results (Console Output) ---")
    print_formatted_results(formatted)

    # Test saving to JSON
    json_output_path = "test_formatted_results.json"
    save_results_to_json(formatted, json_output_path)
    logger.info(f"Test results saved to {json_output_path}. Please verify content.")
    
    # Basic check: results should be sorted by original timeline_start_us.
    # clip_A (timeline 0) should come before clip_B (timeline 5_000_000)
    if formatted and len(formatted) == 2:
        assert formatted[0]["编辑器ID"] == "clip_A"
        assert formatted[1]["编辑器ID"] == "clip_B"
        logger.info("Sorting test passed: Results are in correct order based on timeline_start_us.")
    else:
        logger.error(f"Sorting test failed or unexpected number of results: {len(formatted)}")

    logger.info("output_formatter.py test run finished.")
