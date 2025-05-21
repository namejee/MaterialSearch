# Video Clip Matching System

## Overview

This system is designed to identify and match video segments from a CapCut editing project's source clips to a main target video (often referred to as the "original plate" or "原片"). It analyzes the content of video frames using CLIP models to find semantically similar sequences, even if the clips have undergone transformations like color grading or minor edits.

The primary script for this functionality is `main_matcher.py`.

## Features

*   **CapCut Project Parsing**: Ingests `draft_content.json` files from CapCut projects to understand the source clips used.
*   **Target Video Preprocessing**: Extracts and stores frame features from the main target video using CLIP models. This is a one-time process per video unless forced.
*   **Source Clip Processing**: Samples frames from each source clip identified in the CapCut project and computes their CLIP features.
*   **Content-Based Matching**: Matches sequences of frames from source clips to the target video based on the cosine similarity of their CLIP features and temporal coherence.
*   **Formatted Output**: Provides detailed match results, including:
    *   Editor IDs of the matched CapCut clips.
    *   Original timeline start times in HH:MM:SS:FF format.
    *   Original start/end times of the source clip segment used in CapCut, in HH:MM:SS:FF.
    *   Start/end times of the matched segment within the target video, in HH:MM:SS:FF.
    *   Average similarity score and number of matched samples.
*   **Configurable Matching**: Offers command-line parameters to adjust matching sensitivity, frame sampling intervals, minimum match length, and time drift tolerance.

## Requirements

*   **Python**: 3.x (Developed and tested with Python 3.10+)
*   **Dependencies**: Key libraries include:
    *   OpenCV (`opencv-python`) for video processing.
    *   PyTorch (`torch`) for deep learning operations.
    *   Transformers (`transformers`) by Hugging Face for CLIP model loading and usage.
    *   NumPy (`numpy`) for numerical operations.
    *   SQLAlchemy (`SQLAlchemy`) for database interaction (storing target video features).
    *   A complete list is available in `requirements.txt` (and `requirements_windows.txt` for Windows-specific needs).
*   **FFmpeg**: Recommended for comprehensive video format support by OpenCV. Ensure FFmpeg binaries are in your system's PATH or accessible to OpenCV.
*   **CLIP Models**: The specified CLIP model (defined in `config.py` as `MODEL_NAME`) will be downloaded automatically from the Hugging Face Hub on the first run if not already cached locally.

## Setup/Installation

1.  **Clone Repository**: If applicable, clone the project repository to your local machine.
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```
2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    For Windows, you might also need to consult `requirements_windows.txt` or install specific packages like PyTorch with CUDA support manually from their official instructions if needed.

3.  **Install FFmpeg**:
    *   **Windows**: You can use the provided `install_ffmpeg.bat` script (if available in the broader project) or download FFmpeg from [their official website](https://ffmpeg.org/download.html) and add its `bin` directory to your system's PATH.
    *   **Linux/macOS**: Install FFmpeg using your system's package manager (e.g., `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS).

4.  **CLIP Model Download**: The first time `main_matcher.py` (or any script utilizing `process_assets.py`) runs, it may take some time to download the specified CLIP model from the Hugging Face Hub. Subsequent runs will use the cached model.

## Configuration (`config.py` and `.env`)

General settings for the underlying asset processing and model selection are managed through `config.py` and an optional `.env` file. Key configurations relevant to this matching system include:

*   `MODEL_NAME`: The Hugging Face identifier for the CLIP model to be used (e.g., `"OFA-Sys/chinese-clip-vit-base-patch16"`).
*   `DEVICE`: The device for running model inference (e.g., `"cuda"`, `"cpu"`, `"auto"`).
*   `SQLALCHEMY_DATABASE_URL`: The connection string for the database where target video features are stored (e.g., `"sqlite:///./instance/assets.db"`).
*   `FRAME_INTERVAL`: Interval in seconds for extracting frames during the target video preprocessing step. This is different from the `--sample_interval_sec` used for source clips during matching.

For more detailed information on general project configuration, please refer to the main project's `README.md` (if this script is part of a larger project).

## How to Run `main_matcher.py`

The script is executed from the command line.

**Usage:**

```bash
python main_matcher.py <draft_json_path> <target_video_path> [options]
```

**Arguments:**

*   `draft_json_path`: (Required) Path to the CapCut `draft_content.json` file.
*   `target_video_path`: (Required) Path to the target "original" video file to match against.
*   `--output_file OUTPUT_FILE`: Optional. Path to save the matching results as a JSON file. If omitted, results are printed to the console.
*   `--force_reprocess_target`: Optional. If set, forces the reprocessing of the target video's features, even if they already exist in the database. Default is `False`.
*   `--sample_interval_sec SAMPLE_INTERVAL_SEC`: Optional. Interval in seconds for sampling frames from the source clips. Default: `0.5`.
*   `--similarity_threshold SIMILARITY_THRESHOLD`: Optional. Cosine similarity threshold for considering two frames as a match. Range [0.0 - 1.0]. Default: `0.7`.
*   `--min_match_sequence_len MIN_MATCH_SEQUENCE_LEN`: Optional. Minimum number of consecutive matched frames required to form a valid matching segment. Default: `3`.
*   `--max_time_drift_sec MAX_TIME_DRIFT_SEC`: Optional. Maximum allowed time difference (drift) in seconds between the expected and actual timestamps of corresponding frames in a sequence. Default: `0.5`.

**Example Command:**

```bash
python main_matcher.py "./project_alpha/draft_content.json" "./raw_footage/original_video_day1.mp4" --output_file "./project_alpha/matches_day1.json" --similarity_threshold 0.75 --min_match_sequence_len 3
```

## Output Format

The script outputs a list of matching segments. If an `--output_file` is specified, this list is saved as a JSON array. Otherwise, it's printed to the console in a readable format.

Each item in the list is a dictionary representing a single matched segment, with the following fields:

*   `"编辑器ID"`: The editor ID of the source clip from the CapCut project (e.g., `segment.id`).
*   `"剪映时间轴起始时间"`: The start time of this source clip on the CapCut project timeline, in HH:MM:SS:FF format.
*   `"剪辑源视频原始起止时间"`: The original start and end times of the segment within its source video file (as defined in CapCut's `source_timerange`), in HH:MM:SS:FF format.
*   `"匹配的原片视频起止时间"`: The start and end times of the segment found in the target (original plate) video, in HH:MM:SS:FF format.
*   `"平均相似度"`: The average cosine similarity of all matched frame pairs in this segment, formatted as a percentage (e.g., "88.76%").
*   `"采样点数量"`: The number of frame pairs that constitute this matched segment.

**Example JSON Output Snippet:**
```json
[
    {
        "编辑器ID": "09B9C446-9B95-4A5E-8C1A-AD2F6AC753E1",
        "剪映时间轴起始时间": "00:00:05:10",
        "剪辑源视频原始起止时间": "00:00:10:00 - 00:00:15:15",
        "匹配的原片视频起止时间": "00:01:30:05 - 00:01:35:20",
        "平均相似度": "92.50%",
        "采样点数量": 10
    },
    // ... more match results
]
```

## Troubleshooting (Optional)

*   **Model Download Issues**: If model downloads fail, check your internet connection and Hugging Face Hub accessibility. Ensure you have enough disk space in the cache directory (usually `~/.cache/huggingface/hub`).
*   **FFmpeg Not Found**: OpenCV might fail to process certain video formats. Ensure FFmpeg is correctly installed and in your system's PATH.
*   **CUDA/GPU Issues**: If using GPU (`DEVICE="cuda"`), ensure NVIDIA drivers, CUDA Toolkit, and a CUDA-compatible PyTorch version are correctly installed. If issues persist, try `DEVICE="cpu"`.
*   **File Path Problems**: Ensure all file paths provided as arguments or in configuration are correct and accessible. Use absolute paths if relative paths cause issues.
*   **Permission Errors**: Check read/write permissions for input files, output directories, and the database location.

## Underlying Technology (Optional)

*   **CLIP Models**: Utilizes Contrastive Language-Image Pre-Training (CLIP) models for generating robust semantic feature embeddings from video frames. These embeddings allow for content-based similarity comparison.
*   **OpenCV**: Leveraged for all video I/O operations, frame extraction, and accessing video metadata like FPS.
*   **SQLAlchemy**: Used for managing a database (SQLite by default) that stores precomputed frame features for the target video, enabling faster subsequent runs.

This documentation provides a comprehensive guide to understanding, setting up, and using the `main_matcher.py` script.
