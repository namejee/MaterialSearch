import cv2
import json
import logging
import os
from datetime import datetime
from typing import Generator, Tuple, Optional

import numpy as np
from sqlalchemy.orm import Session

# Assuming these modules are in the PYTHONPATH or same directory
import database
import process_assets # Mocked or actual
import utils
from config import SQLALCHEMY_DATABASE_URL, ENABLE_CHECKSUM, LOG_LEVEL
from models import DatabaseSession # Correct sessionmaker

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

def _get_db_session() -> Session:
    """Provides a database session."""
    return DatabaseSession()

def _prepare_features_generator(
    raw_generator: Generator[Tuple[float, np.ndarray], None, None]
) -> Generator[Tuple[int, bytes], None, None]:
    """
    Adapts the raw feature generator from process_video to what database.add_video expects.
    It converts frame_time_seconds to int and feature_vector to bytes.
    """
    for frame_time_seconds, feature_vector_np in raw_generator:
        # Convert frame_time_seconds (float) to int.
        # Based on FRAME_INTERVAL being int, these should be whole numbers.
        frame_time_int = int(round(frame_time_seconds))
        yield frame_time_int, feature_vector_np.tobytes()


def preprocess_target_video(video_path: str, force_reprocess: bool = False) -> Optional[Tuple[str, str]]:
    """
    Preprocesses the target video: extracts features, stores them in the database,
    and generates a JSON file with frame timestamps.

    Args:
        video_path: Path to the target video file.
        force_reprocess: If True, reprocesses the video even if already in DB.

    Returns:
        A tuple (path_to_json_file, path_to_db_file) or None if an error occurs.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None

    db_session = _get_db_session()
    try:
        video_already_processed = database.is_video_exist(db_session, video_path)
        logger.info(f"Video {video_path} already processed: {video_already_processed}")

        if force_reprocess and video_already_processed:
            logger.info(f"Force reprocessing enabled. Deleting existing frames for {video_path}.")
            database.delete_video_by_path(db_session, video_path) # This is the correct function
            video_already_processed = False # Reset status
            db_session.commit() # delete_video_by_path commits, but good practice if it didn't
            logger.info(f"Existing frames deleted for {video_path}.")
        
        db_session.close() # Close session after check/delete, reopen for add_video if needed or pass session

        if not video_already_processed:
            logger.info(f"Processing video: {video_path}")
            
            modify_time_float = os.path.getmtime(video_path)
            modify_time_dt = datetime.fromtimestamp(modify_time_float)
            
            checksum = None
            if ENABLE_CHECKSUM:
                checksum = utils.get_file_hash(video_path)
                logger.info(f"Checksum for {video_path}: {checksum}")

            # Get raw features generator
            raw_features_gen = process_assets.process_video(video_path)
            
            # Adapt it for database.add_video
            adapted_features_gen = _prepare_features_generator(raw_features_gen)

            # Add to database
            db_session_add = _get_db_session() # New session for this transaction
            try:
                # The existing database.add_video expects datetime for modify_time
                database.add_video(db_session_add, video_path, modify_time_dt, checksum, adapted_features_gen)
                db_session_add.commit() # add_video uses bulk_save and commits itself.
                logger.info(f"Video features stored in database for {video_path}.")
            except Exception as e:
                logger.error(f"Error processing or storing video features for {video_path}: {e}", exc_info=True)
                db_session_add.rollback()
                return None
            finally:
                db_session_add.close()
        else:
            logger.info(f"Skipping feature extraction for {video_path} as it's already processed.")

        # --- Extract FPS and Save Timestamps JSON ---
        logger.info(f"Generating timestamp JSON for {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file with OpenCV: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        logger.info(f"Extracted FPS for {video_path}: {fps}")

        if fps <= 0:
            logger.warning(f"FPS for {video_path} is {fps}. Using default 25.0 for HMSF conversion if needed by utils.")
            # utils.seconds_to_hmsf handles fps <= 0 by defaulting.

        db_session_get_times = _get_db_session()
        try:
            # Uses the new get_just_frame_times_by_path which returns list[int]
            frame_times_sec_list = database.get_just_frame_times_by_path(db_session_get_times, video_path)
            logger.info(f"Retrieved {len(frame_times_sec_list)} frame timestamps from DB for {video_path}.")
        finally:
            db_session_get_times.close()
            
        if not frame_times_sec_list:
            logger.warning(f"No frame timestamps found in DB for {video_path} after processing. Cannot generate JSON.")
            # This might happen if process_video yields nothing or add_video failed silently (though it should commit).
            # Or if video was "skipped" but had no frames from a previous run.
            # Consider if this should be an error or if an empty JSON is acceptable.
            # For now, returning None as it's unexpected if processing was supposed to happen.
            if not video_already_processed and not force_reprocess : # only error if we just processed it.
                 logger.error(f"No frame timestamps found for a newly processed video: {video_path}")
                 return None


        output_json_path = f"{os.path.splitext(os.path.basename(video_path))[0]}_timestamps.json"
        # Saving in the current working directory. Consider a specific output dir.
        output_json_path = os.path.join(os.getcwd(), output_json_path)


        timestamp_data = {
            "fps": fps,
            "frames": []
        }
        for time_sec_int in frame_times_sec_list:
            time_sec_float = float(time_sec_int) # Convert int from DB back to float for seconds_to_hmsf if it expects float
            hmsf_time = utils.seconds_to_hmsf(time_sec_float, fps)
            timestamp_data["frames"].append({
                "time_sec": time_sec_float,
                "time_hmsf": hmsf_time
            })

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(timestamp_data, f, indent=4)
            logger.info(f"Timestamp JSON file created: {output_json_path}")
        except IOError as e:
            logger.error(f"Failed to write timestamp JSON file {output_json_path}: {e}", exc_info=True)
            return None

        db_file_path = SQLALCHEMY_DATABASE_URL.replace("sqlite:///", "")
        return output_json_path, db_file_path

    except Exception as e:
        logger.error(f"An unexpected error occurred in preprocess_target_video for {video_path}: {e}", exc_info=True)
        return None
    finally:
        if db_session and db_session.is_active:
            db_session.close()


if __name__ == '__main__':
    print("Running basic test for video_preprocessor.py")
    
    # Create a dummy video file for testing
    dummy_video_filename = "dummy_test_video.mp4"
    if not os.path.exists(dummy_video_filename):
        # Create a very small, short mp4 file using opencv if possible, or just a placeholder
        # For simplicity, we'll just create an empty file and expect OpenCV to fail to open it,
        # but process_assets.process_video (mock) will still run.
        # A real test would use a valid small mp4.
        try:
            # Create a tiny dummy mp4 file using opencv
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Use a small frame size and fps to keep file size minimal
            # FRAME_INTERVAL from config is used by mock process_video, e.g., 2 seconds
            # So, a 5-second video would give 3 frames (0, 2, 4)
            # Let's make a 5-second video at 10 FPS, 64x64
            test_fps = 10
            test_duration_sec = 5 # Should give frames at 0, 2, 4 if FRAME_INTERVAL=2
            
            # Check if FRAME_INTERVAL is available from config for more realistic frame count
            try:
                from config import FRAME_INTERVAL
            except ImportError:
                FRAME_INTERVAL = 2 # Default if not found
                print(f"Warning: FRAME_INTERVAL not in config, using default {FRAME_INTERVAL} for dummy video creation.")

            out = cv2.VideoWriter(dummy_video_filename, fourcc, test_fps, (64, 64))
            if out.isOpened():
                for i in range(int(test_duration_sec * test_fps)): # 5 seconds * 10 fps = 50 frames
                    frame = np.zeros((64, 64, 3), dtype=np.uint8) # Black frame
                    out.write(frame)
                out.release()
                print(f"Created dummy video: {dummy_video_filename}")
            else:
                print(f"Failed to create dummy video writer for {dummy_video_filename}. Test might be limited.")
                # Fallback: create empty file
                open(dummy_video_filename, 'a').close()
                print(f"Created empty placeholder file: {dummy_video_filename}. OpenCV will likely fail to open it.")

        except Exception as e:
            open(dummy_video_filename, 'a').close()
            print(f"Error creating dummy video with OpenCV ({e}), created empty placeholder: {dummy_video_filename}")

    # Ensure the database tables are created (idempotent)
    from models import create_tables, engine as model_engine
    create_tables(model_engine) # Uses engine from models.py
    print("Database tables checked/created.")

    print(f"\n--- Test 1: Initial processing of {dummy_video_filename} ---")
    result = preprocess_target_video(dummy_video_filename, force_reprocess=False)
    if result:
        print(f"Processing successful. JSON: {result[0]}, DB: {result[1]}")
        assert os.path.exists(result[0])
    else:
        print("Processing failed.")

    print(f"\n--- Test 2: Processing again (should skip feature extraction) ---")
    result_skipped = preprocess_target_video(dummy_video_filename, force_reprocess=False)
    if result_skipped:
        print(f"Processing successful (skipped). JSON: {result_skipped[0]}, DB: {result_skipped[1]}")
        assert os.path.exists(result_skipped[0])
        if result: assert result_skipped[0] == result[0] # Should produce same JSON path
    else:
        print("Processing failed (skipped run).")

    print(f"\n--- Test 3: Force reprocessing ---")
    result_forced = preprocess_target_video(dummy_video_filename, force_reprocess=True)
    if result_forced:
        print(f"Processing successful (forced). JSON: {result_forced[0]}, DB: {result_forced[1]}")
        assert os.path.exists(result_forced[0])
        if result: assert result_forced[0] == result[0] # Should produce same JSON path
    else:
        print("Processing failed (forced run).")
    
    # Clean up dummy video and json
    if os.path.exists(dummy_video_filename):
        os.remove(dummy_video_filename)
        print(f"Cleaned up {dummy_video_filename}")
    
    # Construct expected JSON path based on dummy_video_filename
    expected_json_path = f"{os.path.splitext(dummy_video_filename)[0]}_timestamps.json"
    expected_json_path = os.path.join(os.getcwd(), expected_json_path)
    if os.path.exists(expected_json_path):
        os.remove(expected_json_path)
        print(f"Cleaned up {expected_json_path}")
    
    print("\nBasic video_preprocessor tests completed.")
    print("Note: Success of feature extraction depends on the mock 'process_assets.process_video'.")
    print("Correctness of FPS and timestamps depends on the dummy video and OpenCV.")
