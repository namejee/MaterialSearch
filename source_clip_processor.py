import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

# Assuming capcut_parser.py and process_assets.py are in the python path
# or in the same directory structure that allows these imports.
try:
    from capcut_parser import ClipInfo
except ImportError:
    # This is a fallback for environments where capcut_parser might not be directly in PYTHONPATH
    # For the current project structure, it should be importable.
    # If running this file standalone for testing, ensure capcut_parser.py is accessible.
    logging.warning("Could not import ClipInfo from capcut_parser. Ensure it's in PYTHONPATH.")
    # Define a dummy ClipInfo if not found, for basic script structure to be valid,
    # though __main__ test would fail more clearly.
    from dataclasses import dataclass
    @dataclass
    class ClipInfo:
        editor_id: str
        source_video_path: str
        source_start_us: int
        source_duration_us: int
        timeline_start_us: int
        target_video_path: Optional[str] = None

try:
    from process_assets import get_image_feature
except ImportError:
    logging.warning("Could not import get_image_feature from process_assets. Ensure it's in PYTHONPATH.")
    # Dummy function if not found, for basic script structure.
    def get_image_feature(frames: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        logging.error("process_assets.get_image_feature mock called! Real implementation needed for actual features.")
        if frames:
            # Return a list of dummy feature vectors of a plausible shape
            return [np.random.rand(512).astype(np.float32) for _ in frames]
        return None

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_source_clip_frames_and_features(
    clip_info: ClipInfo, 
    sample_interval_sec: float
) -> Optional[List[Tuple[float, np.ndarray]]]:
    """
    Extracts frames and their CLIP features from a source video clip segment.

    Args:
        clip_info: ClipInfo object describing the source video segment.
        sample_interval_sec: Interval in seconds at which to sample frames.

    Returns:
        A list of tuples, where each tuple is (timestamp_in_clip_seconds, feature_vector_np_array),
        or None if errors occur. timestamp_in_clip_seconds is relative to the start of the source video,
        not the start of the clip segment.
    """
    if not clip_info.source_video_path or not os.path.exists(clip_info.source_video_path):
        logger.error(f"Source video path not found or invalid: {clip_info.source_video_path}")
        return None

    video = cv2.VideoCapture(clip_info.source_video_path)
    if not video.isOpened():
        logger.error(f"Failed to open video file: {clip_info.source_video_path}")
        return None

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.error(f"Invalid FPS ({fps}) for video: {clip_info.source_video_path}")
        video.release()
        return None

    clip_start_time_sec = clip_info.source_start_us / 1_000_000.0
    clip_duration_sec = clip_info.source_duration_us / 1_000_000.0
    clip_end_time_sec = clip_start_time_sec + clip_duration_sec

    sampled_frames_features: List[Tuple[float, np.ndarray]] = []
    
    logger.info(f"Processing clip: {clip_info.editor_id} from video {clip_info.source_video_path}")
    logger.info(f"Clip segment: {clip_start_time_sec:.2f}s to {clip_end_time_sec:.2f}s (Duration: {clip_duration_sec:.2f}s)")
    logger.info(f"Sampling interval: {sample_interval_sec}s, Video FPS: {fps:.2f}")

    current_sample_time_sec = clip_start_time_sec
    while current_sample_time_sec < clip_end_time_sec:
        # Frame number for the current_sample_time_sec in the overall video
        frame_num = int(round(current_sample_time_sec * fps))
        
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame_data = video.read()

        if ret:
            rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            
            # get_image_feature expects a list of frames
            feature_batch = get_image_feature([rgb_frame]) 
            
            if feature_batch is not None and len(feature_batch) > 0:
                feature_vector = feature_batch[0]
                if feature_vector is not None:
                    # The timestamp stored is relative to the start of the video file (current_sample_time_sec)
                    # not relative to the clip's own start time (current_sample_time_sec - clip_start_time_sec)
                    # This matches common interpretations of frame timestamps.
                    sampled_frames_features.append((current_sample_time_sec, feature_vector))
                    logger.debug(f"Successfully extracted feature for frame at {current_sample_time_sec:.2f}s (frame {frame_num})")
                else:
                    logger.warning(f"Feature extraction returned None for frame at {current_sample_time_sec:.2f}s (frame {frame_num})")
            else:
                logger.warning(f"get_image_feature returned None or empty list for frame at {current_sample_time_sec:.2f}s (frame {frame_num})")
        else:
            # This can happen if current_sample_time_sec is beyond the actual video length,
            # or if frame seeking is not precise.
            logger.warning(f"Failed to read frame at {current_sample_time_sec:.2f}s (frame {frame_num}) from {clip_info.source_video_path}. End of clip or video might be reached.")
            # It might be better to break if ret is False, as subsequent reads might also fail.
            # However, some frames might be sporadically unreadable.
            # For robust behavior, one might try to read the next few frames or stop.
            # For now, just log and continue to the next sample point.

        current_sample_time_sec += sample_interval_sec

    video.release()
    
    if not sampled_frames_features:
        logger.warning(f"No frames were sampled for clip {clip_info.editor_id} from {clip_info.source_video_path}. "
                       f"Check video duration, clip timings, and sample interval.")

    logger.info(f"Finished processing clip {clip_info.editor_id}. Extracted {len(sampled_frames_features)} features.")
    return sampled_frames_features

# Need to import os for path checks
import os

if __name__ == '__main__':
    logger.info("Running test for source_clip_processor.py")

    # This test requires a video file. Create a dummy one if not present.
    dummy_video_path = "test_source_video.mp4"
    
    if not os.path.exists(dummy_video_path):
        logger.info(f"Creating dummy video for testing: {dummy_video_path}")
        # Create a small, short MP4 video using OpenCV
        try:
            frame_width, frame_height = 640, 480
            test_fps = 30.0
            duration_sec = 10 # Create a 10-second video

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(dummy_video_path, fourcc, test_fps, (frame_width, frame_height))
            
            if not writer.isOpened():
                raise IOError(f"Could not open video writer for {dummy_video_path}")

            for i in range(int(duration_sec * test_fps)):
                frame = np.random.randint(0, 256, (frame_height, frame_width, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
            logger.info(f"Dummy video '{dummy_video_path}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create dummy video '{dummy_video_path}': {e}. Test may not run correctly.", exc_info=True)
            # Create an empty file as a fallback to allow the script to run further,
            # though feature extraction will likely fail or return empty.
            open(dummy_video_path, 'a').close()


    if os.path.exists(dummy_video_path):
        # Define a dummy ClipInfo object using the created video
        # This ClipInfo will try to extract from 2s to 7s of the 10s video.
        dummy_clip = ClipInfo(
            editor_id="test_clip_001",
            source_video_path=dummy_video_path,
            source_start_us=2_000_000,  # Start at 2 seconds
            source_duration_us=5_000_000, # Duration of 5 seconds (from 2s to 7s)
            timeline_start_us=0, # Not used by this function
            target_video_path="dummy_target.mp4" # Not used by this function
        )
        
        sample_interval = 0.5  # Sample a frame every 0.5 seconds

        logger.info(f"Attempting to extract features for dummy_clip: {dummy_clip.editor_id}")
        results = extract_source_clip_frames_and_features(dummy_clip, sample_interval)

        if results:
            logger.info(f"Extracted {len(results)} frames with features for {dummy_clip.editor_id}:")
            for timestamp, feature_vector in results:
                logger.info(f"  Timestamp: {timestamp:.2f}s, Feature shape: {feature_vector.shape}")
            # Expected timestamps: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5. (10 frames)
            # (Since current_sample_time_sec < clip_end_time_sec, 7.0s itself might not be included if interval makes it hit 7.0 exactly)
            # clip_end_time_sec = 2.0 + 5.0 = 7.0
            # Loop condition is current_sample_time_sec < 7.0
            # So, 6.5 will be the last. 6.5 + 0.5 = 7.0, which is not < 7.0.
            # Thus, 10 samples are expected.
            if len(results) == 10: # 5.0 seconds duration / 0.5s interval = 10 samples
                 logger.info("Test passed: Correct number of samples extracted.")
            else:
                 logger.error(f"Test failed: Expected 10 samples, got {len(results)}")
        else:
            logger.error(f"Feature extraction failed or returned no results for {dummy_clip.editor_id}.")
        
        # Test with a clip segment that goes beyond actual video duration (if video is shorter than segment)
        # The dummy video is 10s. Let's define a clip from 8s for 5s (i.e., 8s to 13s)
        dummy_clip_overflow = ClipInfo(
            editor_id="test_clip_002_overflow",
            source_video_path=dummy_video_path,
            source_start_us=8_000_000,  # Start at 8 seconds
            source_duration_us=5_000_000, # Duration of 5 seconds (tries to go up to 13s)
            timeline_start_us=0,
            target_video_path="dummy_target.mp4"
        )
        logger.info(f"\nAttempting to extract features for dummy_clip_overflow: {dummy_clip_overflow.editor_id}")
        results_overflow = extract_source_clip_frames_and_features(dummy_clip_overflow, sample_interval)
        if results_overflow:
            logger.info(f"Extracted {len(results_overflow)} frames for {dummy_clip_overflow.editor_id}:")
            for timestamp, feature_vector in results_overflow:
                logger.info(f"  Timestamp: {timestamp:.2f}s, Feature shape: {feature_vector.shape}")
            # Expected: 8.0, 8.5, 9.0, 9.5. (4 frames)
            # Video ends at 10.0s. current_sample_time_sec < 13.0s
            # 9.5s + 0.5s = 10.0s. Frame at 10.0s might be readable if video.read() is inclusive of last frame time.
            # cv2.CAP_PROP_POS_FRAMES usually refers to the next frame to be decoded.
            # If video is 10s long, max frame time is just under 10.0.
            # So 9.5s should be the last successfully read frame.
            if len(results_overflow) == 4:
                 logger.info("Test for overflow clip passed: Correct number of samples extracted.")
            else:
                 logger.error(f"Test for overflow clip failed: Expected 4 samples, got {len(results_overflow)}")
        else:
            logger.error(f"Feature extraction failed or returned no results for {dummy_clip_overflow.editor_id}.")

    else:
        logger.warning(f"Skipping main test execution as dummy video '{dummy_video_path}' could not be created/found.")

    # Clean up dummy video
    # if os.path.exists(dummy_video_path):
    #     os.remove(dummy_video_path)
    #     logger.info(f"Cleaned up dummy video: {dummy_video_path}")
    # Commenting out cleanup to allow inspection of the dummy video if needed.
    
    logger.info("\nsource_clip_processor.py test run finished.")
