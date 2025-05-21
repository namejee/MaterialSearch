import unittest
import json
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from capcut_parser import parse_draft_content_json, CapCutProject, ClipInfo

class TestCapCutParser(unittest.TestCase):

    def setUp(self):
        # Create a dummy json file path for tests that need a file
        self.dummy_json_file_path = "test_draft_content.json"

    def tearDown(self):
        # Clean up the dummy json file if it was created
        if os.path.exists(self.dummy_json_file_path):
            os.remove(self.dummy_json_file_path)

    def _create_dummy_json_file(self, content: dict):
        with open(self.dummy_json_file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f)

    def test_parse_valid_json(self):
        valid_json_content = {
            "materials": {
                "videos": [
                    {"id": "mat_id_1", "path": "/path/to/videoA.mp4", "type": "video"},
                    {"id": "mat_id_2", "path": "/path/to/videoB.mp4", "type": "video"}
                ]
            },
            "tracks": [
                {
                    "id": "track1", "type": "video",
                    "segments": [
                        {
                            "id": "seg1", "common_material_id": "mat_id_1",
                            "source_timerange": {"start": 0, "duration": 500000}, # 0.5s
                            "target_timerange": {"start": 0, "duration": 500000}
                        },
                        {
                            "id": "seg2", "common_material_id": "mat_id_2",
                            "source_timerange": {"start": 100000, "duration": 100000}, # 0.1s (too short)
                            "target_timerange": {"start": 500000, "duration": 100000}
                        },
                        {
                            "id": "seg3", "common_material_id": "mat_id_1",
                            "source_timerange": {"start": 600000, "duration": 250000}, # 0.25s
                            "target_timerange": {"start": 600000, "duration": 250000},
                            "extra_material_info": {"path_in_extra": "/extra/path.mp4"} # Test target_video_path_override
                        }
                    ]
                }
            ]
        }
        self._create_dummy_json_file(valid_json_content)
        
        # Test with target_video_path_override
        target_override_path = "/override/target.mp4"
        project = parse_draft_content_json(self.dummy_json_file_path, target_video_path_override=target_override_path)

        self.assertIsInstance(project, CapCutProject)
        self.assertEqual(len(project.source_clips), 2) # seg2 should be filtered out

        # Check seg1
        clip1 = project.source_clips[0]
        self.assertEqual(clip1.editor_id, "seg1")
        self.assertEqual(clip1.source_video_path, "/path/to/videoA.mp4")
        self.assertEqual(clip1.source_start_us, 0)
        self.assertEqual(clip1.source_duration_us, 500000)
        self.assertEqual(clip1.timeline_start_us, 0)
        self.assertEqual(clip1.target_video_path, target_override_path)

        # Check seg3
        clip2 = project.source_clips[1]
        self.assertEqual(clip2.editor_id, "seg3")
        self.assertEqual(clip2.source_video_path, "/path/to/videoA.mp4") # material_id refers to videoA
        self.assertEqual(clip2.source_start_us, 600000)
        self.assertEqual(clip2.source_duration_us, 250000)
        self.assertEqual(clip2.timeline_start_us, 600000)
        self.assertEqual(clip2.target_video_path, target_override_path)

        self.assertIn("/path/to/videoA.mp4", project.potential_source_video_paths)
        self.assertNotIn("/path/to/videoB.mp4", project.potential_source_video_paths) # Because seg2 was too short
        self.assertIn(target_override_path, project.potential_target_video_paths)


    def test_filtering_short_clips(self):
        # All clips are shorter than 0.2 seconds (200000 us)
        short_clips_content = {
            "materials": {"videos": [{"id": "mat1", "path": "/v.mp4"}]},
            "tracks": [{
                "type": "video", "segments": [
                    {"id": "s1", "common_material_id": "mat1", "source_timerange": {"start": 0, "duration": 199999}},
                    {"id": "s2", "common_material_id": "mat1", "source_timerange": {"start": 0, "duration": 100000}}
                ]
            }]
        }
        self._create_dummy_json_file(short_clips_content)
        project = parse_draft_content_json(self.dummy_json_file_path)
        self.assertEqual(len(project.source_clips), 0)
        self.assertEqual(len(project.potential_source_video_paths), 0)

    def test_empty_json_content(self):
        self._create_dummy_json_file({}) # Empty JSON
        project = parse_draft_content_json(self.dummy_json_file_path)
        self.assertEqual(len(project.source_clips), 0)
        self.assertEqual(len(project.potential_source_video_paths), 0)
        self.assertEqual(len(project.potential_target_video_paths), 0)

    def test_malformed_json_file(self):
        with open(self.dummy_json_file_path, 'w') as f:
            f.write("this is not valid json")
        
        # The parser prints an error and returns an empty project
        project = parse_draft_content_json(self.dummy_json_file_path)
        self.assertEqual(len(project.source_clips), 0)

    def test_file_not_found(self):
        # The parser prints an error and returns an empty project
        project = parse_draft_content_json("non_existent_file.json")
        self.assertEqual(len(project.source_clips), 0)

    def test_missing_materials_or_tracks(self):
        content_no_materials = {"tracks": []}
        self._create_dummy_json_file(content_no_materials)
        project = parse_draft_content_json(self.dummy_json_file_path)
        self.assertEqual(len(project.source_clips), 0)

        content_no_tracks = {"materials": {"videos": []}}
        self._create_dummy_json_file(content_no_tracks)
        project = parse_draft_content_json(self.dummy_json_file_path)
        self.assertEqual(len(project.source_clips), 0)

    def test_missing_fields_in_segment(self):
        # Segments missing crucial fields like 'common_material_id' or 'source_timerange'
        content_missing_fields = {
            "materials": {"videos": [{"id": "mat1", "path": "/v.mp4"}]},
            "tracks": [{
                "type": "video", "segments": [
                    {"id": "s1"}, # Missing common_material_id and source_timerange
                    {"id": "s2", "common_material_id": "mat1"}, # Missing source_timerange
                    {
                        "id": "s3", "common_material_id": "mat1", 
                        "source_timerange": {"start": 0} # Missing duration in source_timerange
                    },
                    { # Valid segment for comparison
                        "id": "s4", "common_material_id": "mat1",
                        "source_timerange": {"start": 0, "duration": 500000},
                        "target_timerange": {"start": 0, "duration": 500000}
                    }
                ]
            }]
        }
        self._create_dummy_json_file(content_missing_fields)
        project = parse_draft_content_json(self.dummy_json_file_path)
        self.assertEqual(len(project.source_clips), 1) # Only s4 should be parsed
        if len(project.source_clips) == 1:
            self.assertEqual(project.source_clips[0].editor_id, "s4")

    def test_no_video_materials(self):
        content = {
            "materials": {"audios": [{"id": "aud1", "path": "/a.mp3"}]}, # No 'videos' key
            "tracks": [{
                "type": "video", "segments": [
                    {"id": "s1", "common_material_id": "aud1", "source_timerange": {"start":0, "duration":500000}}
                ]
            }]
        }
        self._create_dummy_json_file(content)
        project = parse_draft_content_json(self.dummy_json_file_path)
        self.assertEqual(len(project.source_clips), 0) # No video materials to link to

    def test_segment_id_fallback(self):
        # Test if segment.id falls back to segment.material_id if 'id' is not present
        # The current parser uses segment.get('id', segment.get('material_id'))
        # My implementation was: editor_id = segment.get('id', segment.get('material_id'))
        # The provided parser might be: editor_id = segment.get('id') or segment.get('material_id')
        # The provided parser in `capcut_parser.py` for the first task was:
        # editor_id = segment.get('id', segment.get('material_id'))
        # This should be fine.
        content = {
            "materials": {"videos": [{"id": "mat1", "path": "/v.mp4"}]},
            "tracks": [{
                "type": "video", "segments": [
                    {
                        "material_id": "seg_mat_id_1", # No 'id' field, should use material_id
                        "common_material_id": "mat1",
                        "source_timerange": {"start": 0, "duration": 500000},
                        "target_timerange": {"start": 0, "duration": 500000}
                    }
                ]
            }]
        }
        self._create_dummy_json_file(content)
        project = parse_draft_content_json(self.dummy_json_file_path)
        self.assertEqual(len(project.source_clips), 1)
        self.assertEqual(project.source_clips[0].editor_id, "seg_mat_id_1")


if __name__ == '__main__':
    unittest.main()
