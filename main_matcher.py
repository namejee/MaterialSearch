import argparse
import json
import logging
import os
import sys

# Assuming current directory structure allows these imports
# (e.g., all modules are in the same top-level directory or PYTHONPATH is set up)
try:
    from capcut_parser import parse_draft_content_json, CapCutProject, ClipInfo
    from video_preprocessor import preprocess_target_video
    from source_clip_processor import extract_source_clip_frames_and_features
    from clip_matcher import find_matching_segments, MatchResult
    from output_formatter import format_match_results, print_formatted_results, save_results_to_json
    # utils.get_video_fps is used by output_formatter, so direct import here might not be needed.
    # Config is imported for its side effects (e.g. logging level) and potentially some defaults if not overridden by args
    import config 
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure all required Python files are in the correct path.", file=sys.stderr)
    sys.exit(1)

# Basic logging configuration
# The log level from config.py will be used if it's set before this,
# otherwise, INFO is a good default.
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Match video clips from a CapCut project to a target video.")
    parser.add_argument("draft_json_path", help="Path to the CapCut draft_content.json file.")
    parser.add_argument("target_video_path", help="Path to the target video file (原片视频).")
    parser.add_argument("--output_file", help="Optional path to save the results as a JSON file. Prints to console if not provided.")
    parser.add_argument("--force_reprocess_target", action="store_true", default=False,
                        help="Force reprocessing of the target video's features (default: False).")
    parser.add_argument("--sample_interval_sec", type=float, default=0.5,
                        help="Interval in seconds for sampling frames from source clips (default: 0.5s).")
    parser.add_argument("--similarity_threshold", type=float, default=0.7, # Example default
                        help="Cosine similarity threshold for matching frames (default: 0.7).")
    parser.add_argument("--min_match_sequence_len", type=int, default=3, # Example default, problem states 2
                        help="Minimum number of consecutive matched samples to form a valid match (default: 3).")
    parser.add_argument("--max_time_drift_sec", type=float, default=0.5, # Example default
                        help="Maximum allowed time drift (in seconds) between consecutive matched frames in a sequence (default: 0.5s).")
    
    args = parser.parse_args()

    logger.info("Starting main_matcher script with arguments: %s", args)

    # --- 1. Parse CapCut Project ---
    logger.info(f"Parsing CapCut project from: {args.draft_json_path}")
    # Pass target_video_path as override for ClipInfo.target_video_path
    capcut_project = parse_draft_content_json(args.draft_json_path, args.target_video_path)
    if not capcut_project or not capcut_project.source_clips:
        logger.error("Failed to parse CapCut project or no source clips found. Exiting.")
        return
    logger.info(f"Successfully parsed CapCut project. Found {len(capcut_project.source_clips)} source clips.")

    # --- 2. Preprocess Target Video ---
    logger.info(f"Preprocessing target video: {args.target_video_path}")
    preprocess_result = preprocess_target_video(args.target_video_path, args.force_reprocess_target)
    if not preprocess_result:
        logger.error("Failed to preprocess target video. Exiting.")
        return
    
    target_timestamp_json_path, _ = preprocess_result # DB path also returned, not directly used here
    logger.info(f"Target video preprocessed. Timestamp JSON at: {target_timestamp_json_path}")

    target_video_fps: float = 30.0 # Default fallback
    try:
        with open(target_timestamp_json_path, 'r') as f:
            timestamp_data = json.load(f)
            target_video_fps = timestamp_data.get("fps", target_video_fps)
        logger.info(f"Target video FPS loaded from JSON: {target_video_fps}")
    except Exception as e:
        logger.warning(f"Could not load target video FPS from {target_timestamp_json_path}: {e}. Using default {target_video_fps} FPS.", exc_info=True)


    # --- 3. Iterate Through Source Clips and Match ---
    all_match_results: List[MatchResult] = []
    logger.info("Starting processing of source clips...")

    for clip_idx, clip_info in enumerate(capcut_project.source_clips):
        logger.info(f"Processing source clip {clip_idx + 1}/{len(capcut_project.source_clips)}: ID {clip_info.editor_id}, Path {clip_info.source_video_path}")

        # Extract features from the current source clip
        source_frames_with_features = extract_source_clip_frames_and_features(
            clip_info, 
            args.sample_interval_sec
        )

        if not source_frames_with_features:
            logger.warning(f"No features extracted for clip {clip_info.editor_id}. Skipping matching for this clip.")
            continue
        
        logger.info(f"Extracted {len(source_frames_with_features)} features for clip {clip_info.editor_id}. Finding matches...")

        # Find matching segments in the target video
        # Note: Renamed parameters for clarity when passing from clip_info
        clip_match_results = find_matching_segments(
            source_frames_with_features=source_frames_with_features,
            target_video_path=args.target_video_path,
            similarity_threshold=args.similarity_threshold,
            min_match_sequence_len=args.min_match_sequence_len,
            max_time_drift_sec=args.max_time_drift_sec,
            source_clip_editor_id=clip_info.editor_id,
            source_video_path_from_clip=clip_info.source_video_path, 
            source_timeline_start_us_from_clip=clip_info.timeline_start_us
        )

        if clip_match_results:
            logger.info(f"Found {len(clip_match_results)} matches for clip {clip_info.editor_id}.")
            all_match_results.extend(clip_match_results)
        else:
            logger.info(f"No matches found for clip {clip_info.editor_id}.")

    # --- 4. Format and Output Results ---
    if not all_match_results:
        logger.info("No matching segments found for any clips.")
        print("No matching segments found.") # Also print to console for user
        return

    logger.info(f"Total matching segments found: {len(all_match_results)}. Formatting results...")
    
    formatted_results = format_match_results(
        all_match_results, 
        capcut_project, 
        target_video_fps
    )

    if args.output_file:
        logger.info(f"Saving formatted results to: {args.output_file}")
        save_results_to_json(formatted_results, args.output_file)
    else:
        logger.info("Printing formatted results to console.")
        print_formatted_results(formatted_results)

    logger.info("Main_matcher script finished successfully.")


if __name__ == "__main__":
    # Example of how to run from command line (replace with actual paths):
    # python main_matcher.py "/path/to/draft_content.json" "/path/to/target_video.mp4" --output_file "results.json"
    
    # To make this runnable with dummy data for quick structural check,
    # one might need to ensure dummy files exist or mock function calls extensively.
    # The current implementation assumes the modules and their dependencies (like OpenCV, models) are functional.
    
    # A simple check for critical dependencies like FFmpeg is often handled by OpenCV installation itself.
    # If OpenCV video operations fail, it might indicate missing backend support (like FFmpeg).
    # The script doesn't add an explicit FFmpeg check beyond what OpenCV relies on.
    main()
