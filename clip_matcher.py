import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

# Assuming database.py and models.py are accessible in the Python path
# For models.py, we only need DatabaseSession and potentially engine for standalone testing.
# For database.py, we need get_frame_times_features_by_path.
try:
    from models import DatabaseSession
    from database import get_frame_times_features_by_path, get_db
except ImportError:
    logging.warning("Could not import from models or database. Using placeholders for script structure.")
    # Define placeholders if imports fail, to allow script to be parsable
    # This is mainly for isolated development; in the project, these should be importable.
    class DatabaseSession: # type: ignore
        def __init__(self): pass
        def query(self, *args): return self
        def filter(self, *args): return self
        def filter_by(self, **kwargs): return self
        def order_by(self, *args): return self
        def all(self): return []
        def first(self): return None
        def count(self): return 0
        def close(self): pass
        def commit(self): pass
        def rollback(self): pass
        def delete(self, *args, **kwargs): pass
        def bulk_save_objects(self, *args, **kwargs): pass

    def get_frame_times_features_by_path(session, path: str) -> Tuple[List[int], List[bytes]]: # type: ignore
        return [], []
    
    from contextlib import contextmanager
    @contextmanager
    def get_db(): # type: ignore
        yield DatabaseSession()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    source_clip_editor_id: str  # From original ClipInfo, to be passed through
    source_video_path: str      # From original ClipInfo
    source_timeline_start_us: int # From original ClipInfo (for context, not directly used in matching)

    source_clip_start_sec: float  # Start time of the matched segment *within the source video file*
    source_clip_end_sec: float    # End time of the matched segment *within the source video file*
    
    target_video_start_sec: float # Start time of the matched segment in the target video
    target_video_end_sec: float   # End time of the matched segment in the target video
    
    average_similarity: float
    num_matched_samples: int
    
    source_frame_timestamps: List[float] = field(default_factory=list) # Timestamps from source video file
    target_frame_timestamps: List[float] = field(default_factory=list) # Timestamps from target video file


def _cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Computes cosine similarity. Assumes features are L2 normalized."""
    # Ensure features are 1D arrays
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    return np.dot(feat1, feat2)

def find_matching_segments(
    source_frames_with_features: List[Tuple[float, np.ndarray]], # (timestamp_in_source_video_sec, feature_vector)
    target_video_path: str,
    similarity_threshold: float,
    min_match_sequence_len: int,
    max_time_drift_sec: float,
    # Pass through these fields from the original ClipInfo for MatchResult construction
    source_clip_editor_id: str, 
    source_video_path_from_clip: str,
    source_timeline_start_us_from_clip: int
) -> List[MatchResult]:
    
    results: List[MatchResult] = []

    if not source_frames_with_features or len(source_frames_with_features) < min_match_sequence_len:
        logger.info("Not enough source frames to find a match.")
        return results

    # 1. Load Target Video Features
    target_frames_data: List[Tuple[float, np.ndarray]] = []
    with get_db() as db_session:
        try:
            # In database.py, get_frame_times_features_by_path returns (list[int_ts], list[bytes_feat])
            target_timestamps_int, target_features_bytes_list = get_frame_times_features_by_path(db_session, target_video_path)
            if not target_timestamps_int:
                logger.warning(f"No features found in database for target video: {target_video_path}")
                return results

            for ts_int, feat_bytes in zip(target_timestamps_int, target_features_bytes_list):
                # Assuming feature vectors are float32. This needs to match how they are stored.
                # If process_assets.py uses a different dtype, adjust here.
                feature_vector = np.frombuffer(feat_bytes, dtype=np.float32)
                target_frames_data.append((float(ts_int), feature_vector)) # Convert int timestamp to float
            
            logger.info(f"Loaded {len(target_frames_data)} feature frames for target video {target_video_path}.")

        except Exception as e:
            logger.error(f"Error loading target video features for {target_video_path}: {e}", exc_info=True)
            return results
        finally:
            db_session.close() # Ensure session is closed if get_db doesn't handle it (it should)

    if not target_frames_data or len(target_frames_data) < min_match_sequence_len:
        logger.info(f"Not enough target frames in {target_video_path} to find a match.")
        return results

    # 2. Sequence Matching Algorithm (Greedy approach)
    num_source_frames = len(source_frames_with_features)
    num_target_frames = len(target_frames_data)
    
    source_idx = 0
    while source_idx <= num_source_frames - min_match_sequence_len:
        best_match_for_current_source_start: Optional[MatchResult] = None
        
        s_start_timestamp, s_start_feature = source_frames_with_features[source_idx]

        for target_idx in range(num_target_frames):
            t_start_timestamp, t_start_feature = target_frames_data[target_idx]

            # Try to start a match
            initial_similarity = _cosine_similarity(s_start_feature, t_start_feature)
            if initial_similarity < similarity_threshold:
                continue

            # Potential match starts, try to extend it
            current_match_source_frames_timestamps = [s_start_timestamp]
            current_match_target_frames_timestamps = [t_start_timestamp]
            current_match_similarities = [initial_similarity]
            
            # Pointers for extending the match
            last_s_matched_idx = source_idx
            last_t_matched_idx = target_idx
            
            # Iterate through subsequent source frames
            for next_s_idx in range(source_idx + 1, num_source_frames):
                s_next_timestamp, s_next_feature = source_frames_with_features[next_s_idx]
                
                found_next_t_match = False
                # Search for a matching target frame for s_next_feature
                # Search window can be optimized, but for now, scan forward from last_t_matched_idx + 1
                for next_t_idx in range(last_t_matched_idx + 1, num_target_frames):
                    t_next_timestamp, t_next_feature = target_frames_data[next_t_idx]
                    
                    # Time progression check
                    s_time_diff = s_next_timestamp - current_match_source_frames_timestamps[-1]
                    t_time_diff = t_next_timestamp - current_match_target_frames_timestamps[-1]

                    if abs(t_time_diff - s_time_diff) > max_time_drift_sec:
                        # If target frame has drifted too much, it might be valid for a *later* source frame,
                        # but not for the current s_next_timestamp.
                        # If t_time_diff is much larger, we might need to skip some target frames.
                        # If t_time_diff is much smaller, this t_next is too early for s_next.
                        if t_time_diff > s_time_diff + max_time_drift_sec : # t is too far ahead, break inner t loop for current s
                             break 
                        continue # Check next t frame for current s


                    # Similarity check
                    sim = _cosine_similarity(s_next_feature, t_next_feature)
                    if sim >= similarity_threshold:
                        current_match_source_frames_timestamps.append(s_next_timestamp)
                        current_match_target_frames_timestamps.append(t_next_timestamp)
                        current_match_similarities.append(sim)
                        
                        last_s_matched_idx = next_s_idx # This is implicitly next_s_idx
                        last_t_matched_idx = next_t_idx
                        found_next_t_match = True
                        break # Found match for s_next_timestamp, move to next source frame
                
                if not found_next_t_match:
                    break # Cannot extend match with current_s_frame, end this sequence attempt
            
            # After trying to extend, check if the sequence is long enough
            if len(current_match_source_frames_timestamps) >= min_match_sequence_len:
                avg_sim = sum(current_match_similarities) / len(current_match_similarities)
                
                current_result = MatchResult(
                    source_clip_editor_id=source_clip_editor_id,
                    source_video_path=source_video_path_from_clip,
                    source_timeline_start_us=source_timeline_start_us_from_clip,
                    source_clip_start_sec=current_match_source_frames_timestamps[0],
                    source_clip_end_sec=current_match_source_frames_timestamps[-1],
                    target_video_start_sec=current_match_target_frames_timestamps[0],
                    target_video_end_sec=current_match_target_frames_timestamps[-1],
                    average_similarity=avg_sim,
                    num_matched_samples=len(current_match_source_frames_timestamps),
                    source_frame_timestamps=current_match_source_frames_timestamps,
                    target_frame_timestamps=current_match_target_frames_timestamps
                )
                
                # Greedy choice: if this match is better (longer or more similar) than previous ones starting at source_idx
                if best_match_for_current_source_start is None or \
                   len(current_match_source_frames_timestamps) > best_match_for_current_source_start.num_matched_samples or \
                   (len(current_match_source_frames_timestamps) == best_match_for_current_source_start.num_matched_samples and avg_sim > best_match_for_current_source_start.average_similarity):
                    best_match_for_current_source_start = current_result

        if best_match_for_current_source_start:
            results.append(best_match_for_current_source_start)
            # Advance source_idx past the frames included in this match
            source_idx += best_match_for_current_source_start.num_matched_samples
        else:
            source_idx += 1 # No match found starting at this source_idx, move to next source frame

    return results


if __name__ == '__main__':
    logger.info("Running basic test for clip_matcher.py")

    # Mock data for testing
    # Source frames: (timestamp_sec, feature_vector)
    # Features are simple 1D arrays for easy dot product. Assume normalized.
    mock_source_frames = [
        (0.0, np.array([1.0, 0.0, 0.0])), # S0
        (0.5, np.array([0.0, 1.0, 0.0])), # S1
        (1.0, np.array([0.0, 0.0, 1.0])), # S2
        (1.5, np.array([0.7, 0.7, 0.0])), # S3
        (2.0, np.array([0.0, 0.7, 0.7])), # S4
        (2.5, np.array([0.6, 0.6, 0.6])), # S5 - unmatched
        (3.0, np.array([1.0, 0.0, 0.0])), # S6 (like S0)
        (3.5, np.array([0.0, 1.0, 0.0])), # S7 (like S1)
    ]

    # Mock target frames data (as if loaded from DB)
    # (timestamp_sec, feature_vector)
    mock_target_data_db = [
        (10.0, np.array([0.9, 0.1, 0.0])), # T0 - matches S0
        (10.5, np.array([0.1, 0.9, 0.0])), # T1 - matches S1
        (11.0, np.array([0.1, 0.1, 0.9])), # T2 - matches S2
        # Gap / noise
        (12.0, np.array([0.5, 0.5, 0.5])), # T3
        (12.5, np.array([0.6, 0.7, 0.0])), # T4 - matches S3 (0.7*0.6 + 0.7*0.7 = 0.42 + 0.49 = 0.91)
        (13.0, np.array([0.1, 0.6, 0.6])), # T5 - matches S4 (0.7*0.6 + 0.7*0.6 = 0.42 + 0.42 = 0.84)
        # Another sequence
        (20.0, np.array([1.0, 0.0, 0.0])), # T6 - matches S6 (and S0)
        (20.5, np.array([0.0, 1.0, 0.0])), # T7 - matches S7 (and S1)
        (21.0, np.array([0.0, 0.0, 1.0])), # T8 - (no S match here)
    ]

    # Override database calls with mock data
    original_get_db = get_db # Save original
    
    @contextmanager
    def mock_get_db_context(): # type: ignore
        class MockSession:
            def close(self): pass
        yield MockSession()

    def mock_get_frame_times_features_by_path(session, path: str) -> Tuple[List[int], List[bytes]]: # type: ignore
        logger.info(f"Mock DB: Requesting features for target path {path}")
        # Convert mock_target_data_db to the expected DB output format
        timestamps_int = [int(t) for t, f in mock_target_data_db] # Timestamps are int in DB
        features_bytes_list = [f.astype(np.float32).tobytes() for t, f in mock_target_data_db]
        return timestamps_int, features_bytes_list

    # Apply mocks
    get_frame_times_features_by_path_orig = get_frame_times_features_by_path # if it was imported directly
    
    # Monkey patch: replace the global 'get_db' and 'get_frame_times_features_by_path'
    # This is a bit tricky if they are imported as `from database import get_db` vs `import database`
    # For this test, let's assume we can patch them where they are defined or used.
    # The current structure uses `from database import get_frame_times_features_by_path, get_db`
    # So, we need to patch those specific imported names in this module's scope.
    
    # This is simplified. A more robust way would be to use unittest.mock.patch
    # For now, we will rely on the fact that the functions are defined in this file if imports fail.
    # If imports succeed, this test won't use the mock DB functions correctly without more advanced patching.
    # Let's assume the placeholder functions at the top are active for this __main__ block
    # by simulating an ImportError for the real DB functions.
    
    # To ensure mocks are used, we would typically do:
    # import clip_matcher
    # clip_matcher.get_db = mock_get_db_context
    # clip_matcher.get_frame_times_features_by_path = mock_get_frame_times_features_by_path
    # Then call clip_matcher.find_matching_segments(...)
    # For simplicity here, if the real DB functions are imported, this test will try to use the actual DB.
    # We'll proceed as if the mock setup at the top (placeholders) is sufficient for a basic run.
    # The critical part for testing is the algorithm itself, not the DB interaction here.

    logger.info("Using potentially mocked DB functions for test.")
    # Replace the database access functions with mocks for testing
    _original_get_frame_times_features_by_path = get_frame_times_features_by_path
    _original_get_db = get_db
    
    # Monkey patch the functions in the current module's scope
    globals()['get_frame_times_features_by_path'] = mock_get_frame_times_features_by_path
    globals()['get_db'] = mock_get_db_context


    test_results = find_matching_segments(
        source_frames_with_features=mock_source_frames,
        target_video_path="dummy_target.mp4",
        similarity_threshold=0.8, # High threshold for these simple vectors
        min_match_sequence_len=2,
        max_time_drift_sec=0.2,  # Max 0.2s drift from expected interval (0.5s)
        source_clip_editor_id="clip1",
        source_video_path_from_clip="/path/to/source.mp4",
        source_timeline_start_us_from_clip=0
    )

    logger.info(f"Found {len(test_results)} matching segments:")
    for res_idx, res in enumerate(test_results):
        logger.info(f"  Match {res_idx + 1}:")
        logger.info(f"    Source Clip ID: {res.source_clip_editor_id}")
        logger.info(f"    Source Path: {res.source_video_path}")
        logger.info(f"    Source Segment: {res.source_clip_start_sec:.2f}s - {res.source_clip_end_sec:.2f}s "
                    f"(Timestamps: {res.source_frame_timestamps})")
        logger.info(f"    Target Segment: {res.target_video_start_sec:.2f}s - {res.target_video_end_sec:.2f}s "
                    f"(Timestamps: {res.target_frame_timestamps})")
        logger.info(f"    Num Samples: {res.num_matched_samples}")
        logger.info(f"    Avg Similarity: {res.average_similarity:.3f}")

    # Restore original functions if they were patched globally and directly
    globals()['get_frame_times_features_by_path'] = _original_get_frame_times_features_by_path
    globals()['get_db'] = _original_get_db


    # Expected results from the test data:
    # Match 1: S0,S1,S2 with T0,T1,T2. (Timestamps: S[0,0.5,1.0], T[10,10.5,11.0]) Sim ~0.9
    # source_idx advances by 3. Next search starts at S3.
    # Match 2: S3,S4 with T4,T5. (Timestamps: S[1.5,2.0], T[12.5,13.0]) Sim ~0.87
    # source_idx advances by 2. Next search starts at S5. S5 does not match anything.
    # source_idx advances by 1. Next search starts at S6.
    # Match 3: S6,S7 with T6,T7. (Timestamps: S[3.0,3.5], T[20.0,20.5]) Sim ~1.0

    # Check if results roughly match expectations
    assert len(test_results) == 3
    if len(test_results) == 3:
        assert test_results[0].num_matched_samples == 3 
        assert test_results[0].source_frame_timestamps == [0.0, 0.5, 1.0]
        assert test_results[0].target_frame_timestamps == [10.0, 10.5, 11.0]
        
        assert test_results[1].num_matched_samples == 2
        assert test_results[1].source_frame_timestamps == [1.5, 2.0]
        assert test_results[1].target_frame_timestamps == [12.5, 13.0]

        assert test_results[2].num_matched_samples == 2
        assert test_results[2].source_frame_timestamps == [3.0, 3.5]
        assert test_results[2].target_frame_timestamps == [20.0, 20.5]
        logger.info("Test assertions passed for expected matches.")
    else:
        logger.error(f"Test assertions failed: Expected 3 matches, got {len(test_results)}")

    logger.info("clip_matcher.py test run finished.")
