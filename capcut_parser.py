import json
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional

@dataclass
class ClipInfo:
    editor_id: str
    source_video_path: str
    source_start_us: int
    source_duration_us: int
    timeline_start_us: int
    target_video_path: Optional[str] = None

@dataclass
class CapCutProject:
    source_clips: List[ClipInfo] = field(default_factory=list)
    potential_source_video_paths: Set[str] = field(default_factory=set)
    potential_target_video_paths: Set[str] = field(default_factory=set)

def parse_draft_content_json(json_file_path: str, target_video_path_override: Optional[str] = None) -> CapCutProject:
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        # Consider raising a custom error or returning an empty project
        print(f"Error: JSON file not found at {json_file_path}")
        return CapCutProject()
    except json.JSONDecodeError:
        # Consider raising a custom error or returning an empty project
        print(f"Error: Could not decode JSON from {json_file_path}")
        return CapCutProject()

    project = CapCutProject()
    video_materials_map: Dict[str, str] = {}

    # Extract Video Materials
    if 'materials' in data and 'videos' in data['materials']:
        for video_material in data['materials'].get('videos', []):
            if 'id' in video_material and 'path' in video_material:
                video_materials_map[video_material['id']] = video_material['path']
                # Assuming any video in materials could be a source
                project.potential_source_video_paths.add(video_material['path'])

    # Extract Clips
    for track in data.get('tracks', []):
        # Assuming 'video' tracks contain the relevant clips.
        # CapCut uses different types like "video", "audio", "text".
        # We might need to be more specific if other track types also use video segments.
        if track.get('type') == 'video' or True: # Process all tracks initially, then filter segments by type
            for segment in track.get('segments', []):
                # Ensure segment is a video clip. CapCut might use 'material_type' or infer from 'common_material_id'
                # For now, we assume if common_material_id points to a video, it's a video segment.
                # A more robust check could be segment.get('material_type') == 'video_segment' or similar
                
                common_material_id = segment.get('common_material_id')
                if not common_material_id or common_material_id not in video_materials_map:
                    # This segment might not be a standard video clip linked to materials.videos
                    # or its material definition is missing.
                    continue

                editor_id = segment.get('id', segment.get('material_id')) # Use 'id' if available, else 'material_id'
                if not editor_id:
                    # Skip if no suitable ID can be found
                    continue

                source_video_path = video_materials_map[common_material_id]
                
                source_timerange = segment.get('source_timerange')
                if not source_timerange or 'start' not in source_timerange or 'duration' not in source_timerange:
                    # Skip if timing information is incomplete
                    continue
                
                source_start_us = source_timerange['start']
                source_duration_us = source_timerange['duration']

                target_timerange = segment.get('target_timerange')
                if not target_timerange or 'start' not in target_timerange:
                    # Skip if timeline start is missing
                    continue
                timeline_start_us = target_timerange['start']

                # Handle target_video_path
                current_target_video_path = target_video_path_override
                if current_target_video_path:
                    project.potential_target_video_paths.add(current_target_video_path)

                clip_info = ClipInfo(
                    editor_id=editor_id,
                    source_video_path=source_video_path,
                    source_start_us=source_start_us,
                    source_duration_us=source_duration_us,
                    timeline_start_us=timeline_start_us,
                    target_video_path=current_target_video_path
                )
                
                # Filter clips by duration
                if clip_info.source_duration_us >= 200000: # 0.2 seconds
                    project.source_clips.append(clip_info)


    # Populate potential_source_video_paths from the actually used clips
    # The previous addition was speculative, this is more accurate.
    project.potential_source_video_paths = {clip.source_video_path for clip in project.source_clips}

    return project

if __name__ == '__main__':
    # Basic test with a dummy draft_content.json structure
    # Create a dummy json file for testing
    dummy_json_content = {
        "materials": {
            "videos": [
                { "id": "video_id_1", "path": "/path/to/source_video_A.mp4", "type": "video" },
                { "id": "video_id_2", "path": "/path/to/source_video_B.mp4", "type": "video" },
                { "id": "video_id_3", "path": "/path/to/source_video_C_short.mp4", "type": "video" }
            ]
        },
        "tracks": [
            {
                "id": "track_id_1",
                "type": "video",
                "segments": [
                    {
                        "id": "segment_id_1",
                        "material_id": "segment_id_1_mat",
                        "common_material_id": "video_id_1",
                        "source_timerange": { "start": 0, "duration": 5000000 }, # 5s
                        "target_timerange": { "start": 1000000, "duration": 5000000 }
                    },
                    {
                        "id": "segment_id_2",
                        "common_material_id": "video_id_2",
                        "source_timerange": { "start": 1000000, "duration": 250000 }, # 0.25s
                        "target_timerange": { "start": 6000000, "duration": 250000 }
                    },
                    {
                        "id": "segment_id_3", # Too short
                        "common_material_id": "video_id_3",
                        "source_timerange": { "start": 0, "duration": 100000 }, # 0.1s
                        "target_timerange": { "start": 7000000, "duration": 100000 }
                    },
                    {
                        "id": "segment_id_4_no_timing", # Missing source_timerange.duration
                        "common_material_id": "video_id_1",
                        "source_timerange": { "start": 0 },
                        "target_timerange": { "start": 8000000, "duration": 1000000 }
                    },
                     {
                        "id": "segment_id_5_no_common_id", 
                        "source_timerange": { "start": 0, "duration": 5000000 },
                        "target_timerange": { "start": 9000000, "duration": 5000000 }
                    }
                ]
            },
            { # Non-video track, should be ignored if strict type checking is applied
                "id": "track_id_2",
                "type": "audio",
                 "segments": [
                    { # This segment should be ignored based on track type or lack of video material linkage
                        "id": "audio_segment_1",
                        "material_id": "audio_mat_1",
                        "common_material_id": "some_audio_id_not_in_videos", # Not in video_materials_map
                        "source_timerange": { "start": 0, "duration": 5000000 },
                        "target_timerange": { "start": 1000000, "duration": 5000000 }
                    }]
            }
        ]
    }
    dummy_file_path = "dummy_draft_content.json"
    with open(dummy_file_path, 'w', encoding='utf-8') as f:
        json.dump(dummy_json_content, f)

    print(f"Attempting to parse dummy file: {dummy_file_path}")
    project_data = parse_draft_content_json(dummy_file_path, target_video_path_override="/path/to/target_video.mp4")

    print(f"\nParsed CapCut Project:")
    print(f"  Potential Source Video Paths: {project_data.potential_source_video_paths}")
    print(f"  Potential Target Video Paths: {project_data.potential_target_video_paths}")
    print(f"  Source Clips ({len(project_data.source_clips)}):")
    for clip in project_data.source_clips:
        print(f"    - Clip ID: {clip.editor_id}")
        print(f"      Source Path: {clip.source_video_path}")
        print(f"      Source Start (us): {clip.source_start_us}, Duration (us): {clip.source_duration_us}")
        print(f"      Timeline Start (us): {clip.timeline_start_us}")
        print(f"      Target Path: {clip.target_video_path}")

    # Expected: 2 clips (segment_id_1, segment_id_2)
    # segment_id_3 is too short
    # segment_id_4 has incomplete timing
    # segment_id_5 has no common_material_id linking to a known video
    
    # Clean up dummy file
    import os
    os.remove(dummy_file_path)
    print(f"\nCleaned up dummy file: {dummy_file_path}")

    # Test with non-existent file
    print("\nAttempting to parse non-existent file:")
    project_data_non_existent = parse_draft_content_json("non_existent_draft.json")
    print(f"  Source Clips: {len(project_data_non_existent.source_clips)}")

    # Test with invalid json file
    invalid_json_path = "invalid_draft_content.json"
    with open(invalid_json_path, 'w') as f:
        f.write("{'this is not valid json':,}")
    print("\nAttempting to parse invalid JSON file:")
    project_data_invalid_json = parse_draft_content_json(invalid_json_path)
    print(f"  Source Clips: {len(project_data_invalid_json.source_clips)}")
    os.remove(invalid_json_path)
    print(f"Cleaned up invalid JSON file: {invalid_json_path}")

print("capcut_parser.py created and basic test structure included.")
