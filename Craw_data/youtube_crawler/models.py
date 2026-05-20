"""
Data models cho YouTube Audio Crawler Pipeline.
Định nghĩa các dataclass dùng chung giữa các phase.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class VideoInfo:
    """Thông tin 1 video YouTube."""
    channel_name: str
    title: str
    url: str


@dataclass
class WordTimestamp:
    """Timestamp cho 1 từ trong transcript."""
    word: str
    start_time: float  # seconds
    end_time: float    # seconds


@dataclass
class TranscriptionResult:
    """Kết quả transcription của 1 file audio."""
    wav_path: str
    full_text: str
    word_timestamps: List[WordTimestamp]
    duration: float


@dataclass
class SegmentInfo:
    """Thông tin 1 segment audio đã cắt."""
    output_path: str       # Path to segment WAV (e.g., uuid1_00001.wav)
    transcript: str        # Text content of segment
    duration: float        # Duration in seconds
    source_file: str       # Original source WAV
    start_time: float      # Start position in source
    end_time: float        # End position in source


@dataclass
class StatsInfo:
    """Thống kê cho mỗi folder Step_X."""
    total_files: int
    total_duration_seconds: float
    avg_duration_seconds: float
