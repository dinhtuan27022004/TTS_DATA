"""
Phase 4: Transcription
Script chạy Phase 4 - Transcribe audio tiếng Việt bằng nvidia/parakeet-ctc-0.6b-vi.

Cách dùng:
    cd D:\CO_2026\TTS-DATA
    python Craw_data/run_phase4.py
"""

import sys
import os

# Thêm thư mục cha vào path để import module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from youtube_crawler.transcriber import Transcriber


def main():
    print("=== Phase 4: Transcription (Parakeet CTC) ===")

    transcriber = Transcriber(
        input_dir="Youtube_Data/Step_1",
        model_name="nvidia/parakeet-ctc-0.6b-vi"
    )

    results = transcriber.transcribe_all()

    print(f"\n=== Hoàn thành ===")
    print(f"Số files đã transcribe mới: {len(results)}")


if __name__ == "__main__":
    main()
