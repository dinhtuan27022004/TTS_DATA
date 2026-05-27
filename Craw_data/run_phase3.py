"""
Phase 3: Music Removal
Script chạy Phase 3 - Loại bỏ nhạc nền bằng Facebook Demucs.

Cách dùng:
    cd D:\CO_2026\TTS-DATA
    python Craw_data/run_phase3.py
"""

import sys
import os

# Thêm thư mục cha vào path để import module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from youtube_crawler.music_remover import MusicRemover


def main():
    print("=== Phase 3: Music Removal (Demucs) ===")

    remover = MusicRemover(
        input_dir="Craw_data/Youtube_Data/Step_0",
        output_dir="Craw_data/Youtube_Data/Step_1"
    )

    processed = remover.process_all()

    print(f"\n=== Hoàn thành ===")
    print(f"Số files đã xử lý mới: {len(processed)}")


if __name__ == "__main__":
    main()
