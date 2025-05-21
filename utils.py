import hashlib
import logging
import platform
import subprocess

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from config import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
register_heif_opener()


def get_hash(bytesio):
    """
    计算字节流的 hash
    :param bytesio: bytes 或 BytesIO
    :return: string, 十六进制字符串
    """
    _hash = hashlib.sha1()
    if type(bytesio) is bytes:
        _hash.update(bytesio)
        return _hash.hexdigest()
    try:
        while data := bytesio.read(1048576):
            _hash.update(data)
    except Exception as e:
        logger.error(f"计算hash出错：{bytesio} {repr(e)}")
        return None
    bytesio.seek(0)  # 归零，用于后续写入文件
    return _hash.hexdigest()


def get_string_hash(string):
    """
    计算字符串hash
    :param string: string, 字符串
    :return: string, 十六进制字符串
    """
    _hash = hashlib.sha1()
    _hash.update(string.encode("utf8"))
    return _hash.hexdigest()


def get_file_hash(file_path):
    """
    计算文件的哈希值
    :param file_path: string, 文件路径
    :return: string, 十六进制哈希值，或 None（文件读取错误）
    """
    _hash = hashlib.sha1()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(1048576):
                _hash.update(chunk)
        return _hash.hexdigest()
    except Exception as e:
        logger.error(f"计算文件hash出错：{file_path} {repr(e)}")
        return None


def softmax(x):
    """
    计算softmax，使得每一个元素的范围都在(0,1)之间，并且所有元素的和为1。
    softmax其实还有个temperature参数，目前暂时不用。
    :param x: [float]
    :return: [float]
    """
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores)


def format_seconds(seconds):
    """
    将秒数转成时分秒格式
    :param seconds: int, 秒数
    :return: "时:分:秒"
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def crop_video(input_file, output_file, start_time, end_time):
    """
    调用ffmpeg截取视频片段
    :param input_file: 要截取的文件路径
    :param output_file: 保存文件路径
    :param start_time: int, 开始时间，单位为秒
    :param end_time: int, 结束时间，单位为秒
    :return: None
    """
    cmd = 'ffmpeg'
    if platform.system() == 'Windows':
        cmd += ".exe"
    command = [
        cmd,
        '-ss', format_seconds(start_time),
        '-to', format_seconds(end_time),
        '-i', input_file,
        '-c:v', 'copy',
        '-c:a', 'copy',
        output_file
    ]
    logger.info("Crop video:", " ".join(command))
    subprocess.run(command)


def resize_image_with_aspect_ratio(image_path, target_size, convert_rgb=False):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)  # 根据 EXIF 信息自动旋转图像
    if convert_rgb:
        image = image.convert('RGB')
    # 计算调整后图像的目标大小及长宽比
    width, height = image.size
    aspect_ratio = width / height
    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height
    # 计算调整后图像的实际大小
    if target_aspect_ratio < aspect_ratio:
        # 以目标宽度为准进行调整
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # 以目标高度为准进行调整
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    # 调整图像的大小
    resized_image = image.resize((new_width, new_height))
    return resized_image


def seconds_to_hmsf(seconds: float, fps: float) -> str:
    """Converts seconds to HH:MM:SS:FF format."""
    if fps <= 0:
        # Avoid division by zero or negative FPS, default to a common video FPS if invalid
        logger.warning(f"Invalid FPS value {fps} received, defaulting to 25.0 for HMSF conversion.")
        fps = 25.0 
    
    total_frames_exact = seconds * fps
    # It's common to take the floor of total_frames for frame number calculation
    # e.g. second 0.0 to (1/fps - epsilon) is frame 0
    
    ss_float = seconds
    
    hh = int(ss_float // 3600)
    mm = int((ss_float % 3600) // 60)
    ss = int(ss_float % 60)
    
    # Frame number is the number of full frames that have passed
    # For second 1.0, if FPS is 30, it's the 0th frame of second 1, or 30th frame of video.
    # The "FF" part usually means the frame index within the current second.
    # (seconds - int(seconds)) * fps gives the frame count into the current second.
    frame_within_second = int(round((ss_float - int(ss_float)) * fps))

    # Ensure frame number doesn't exceed fps-1, could happen due to rounding if ss_float is x.99999...
    if frame_within_second >= fps : # If calculated frame is, say, 30 for 30fps, it should be frame 29.
        frame_within_second = int(fps -1) if fps >0 else 0
        # Or, if it implies rolling over to the next second:
        # ss += 1 # and then recalculate hh, mm, ss, frame_within_second = 0
        # For simplicity, capping at fps-1 is common for HH:MM:SS:FF representation.

    return f"{hh:02d}:{mm:02d}:{ss:02d}:{frame_within_second:02d}"


def get_video_fps(video_path: str) -> float | None:
    """
    Retrieves the FPS of a video file using OpenCV.

    Args:
        video_path: Path to the video file.

    Returns:
        The FPS of the video as a float, or None if an error occurs
        (e.g., file not found, cannot open, invalid FPS).
    """
    if not os.path.exists(video_path):
        logger.error(f"get_video_fps: Video file not found at {video_path}")
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"get_video_fps: Failed to open video file: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps is None or fps <= 0:
            logger.warning(f"get_video_fps: Invalid or zero FPS ({fps}) detected for video: {video_path}")
            return None
            
        return fps
    except Exception as e:
        logger.error(f"get_video_fps: An exception occurred while trying to get FPS for {video_path}: {e}", exc_info=True)
        return None

# Need to import os and cv2 for get_video_fps
import os
import cv2 # Assuming opencv-python is installed
