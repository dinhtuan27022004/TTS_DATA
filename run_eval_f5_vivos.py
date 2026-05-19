"""
Script đánh giá F5-TTS V0 trên dataset VIVOS.

Cách chạy:
    python run_eval_f5_vivos.py

Checkpoint và vocab sẽ tự động tải từ HuggingFace nếu chưa có ở local:
    - Model: hf://hynt/F5-TTS-Vietnamese-100h/model_500000.pt
    - Vocab: hf://hynt/F5-TTS-Vietnamese-100h/vocab.txt
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from components.tts.F5_V0 import F5TTSVietnamese
from evaluate.evaluator import TTSEvaluator

# ============================================================
# CẤU HÌNH
# ============================================================

# Dataset path
DATASET_PATH = os.path.join(PROJECT_ROOT, "Processed_DATA", "VIVOS")

# Model name cho báo cáo
MODEL_NAME = "F5-TTS-V0"

# ============================================================


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return

    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_PATH}")
    print("Checkpoint và vocab sẽ tự động tải nếu chưa có.")
    print("=" * 60)

    # Khởi tạo F5-TTS model (auto download checkpoint + vocab)
    model = F5TTSVietnamese(
        vocoder_name="vocos",
        speed=1.0,
    )

    # Khởi tạo evaluator
    evaluator = TTSEvaluator()

    # Chạy đánh giá
    print("\nBắt đầu đánh giá...")
    report = evaluator.evaluate(
        dataset_path=DATASET_PATH,
        tts_model=model,
        model_name=MODEL_NAME,
        force=False,
        save_audio=True,
    )

    # In kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"Số mẫu đánh giá: {len(report.results)}")
    print(f"CSV: {report.csv_path}")
    print(f"Biểu đồ: {report.chart_paths}")
    print("\nSummary Statistics:")
    for metric, stats in report.summary_statistics.items():
        print(f"  {metric:>15}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}")


if __name__ == "__main__":
    main()
