"""
Phase 4: Transcription
Script chạy Phase 4 - Transcribe audio tiếng Việt bằng Whisper Large v3.

Cách dùng:
    cd D:/CO_2026/TTS-DATA
    python Craw_data/run_phase4.py
"""

import sys
import os

# Thêm thư mục gốc TTS_DATA vào path để có thể import thư mục components
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from youtube_crawler.transcriber import Transcriber


def main():
    print("=== Phase 4: Transcription (Whisper Large v3) ===")

    transcriber = Transcriber(
        input_dir="Craw_data/Youtube_Data/Step_1",
        model_name="large-v3",
        language="vi",
        beam_size=5,
        num_workers=1,
    )

    results = transcriber.transcribe_all()

    print(f"\n=== Hoàn thành ===")
    print(f"Số files đã transcribe mới: {len(results)}")


if __name__ == "__main__":
    main()
