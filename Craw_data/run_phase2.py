"""
Phase 2: Audio Download
Script chạy Phase 2 - Tải audio từ YouTube dưới dạng WAV.

Cách dùng:
    cd D:\CO_2026\TTS-DATA
    python Craw_data/run_phase2.py
"""

import sys
import os

# Thêm thư mục cha vào path để import module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from youtube_crawler.downloader import AudioDownloader


def main():
    print("=== Phase 2: Audio Download ===")

    downloader = AudioDownloader(
        urls_excel_path="Craw_data/Youtube_Data/video_urls.xlsx",
        output_dir="Craw_data/Youtube_Data/Step_0"
    )

    mapping = downloader.download_all()

    print(f"\n=== Hoàn thành ===")
    print(f"Tổng số files đã tải: {len(mapping)}")


if __name__ == "__main__":
    main()
