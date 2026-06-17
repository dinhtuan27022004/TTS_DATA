"""
Phase 5: Audio Segmentation
Script chạy Phase 5 - Cắt audio thành đoạn 3-7 giây kèm transcript.

Cách dùng:
    cd D:\CO_2026\TTS-DATA
    python Craw_data/run_phase5.py
"""

import sys
import os

# Thêm thư mục cha vào path để import module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from youtube_crawler.segmenter import AudioSegmenter


def main():
    print("=== Phase 5: Audio Segmentation ===")

    segmenter = AudioSegmenter(
        input_dir="Craw_data/Youtube_Data/Step_1",
        output_dir="Craw_data/Youtube_Data/Step_2",
        min_duration=5.0,
        max_duration=15.0
    )

    segments = segmenter.segment_all()

    print(f"\n=== Hoàn thành ===")
    print(f"Số segments mới tạo: {len(segments)}")


if __name__ == "__main__":
    main()
