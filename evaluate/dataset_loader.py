"""
DatasetLoader - Module tải và parse dataset cho hệ thống đánh giá TTS.

Chịu trách nhiệm:
- Parse file metadata.csv (pipe-separated: relative_path|text|duration)
- Trả về danh sách EvalSample
- Kiểm tra sự tồn tại của file audio
- Tải audio dưới dạng numpy array + sample rate
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np

from evaluate.models import EvalSample

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Tải và parse dataset TTS từ thư mục có cấu trúc chuẩn.

    Dataset cần có cấu trúc:
        dataset_path/
            metadata.csv        (pipe-separated: relative_path|text|duration)
            wavs/
                file1.wav
                file2.wav
                ...
    """

    def load_dataset(self, dataset_path: str) -> List[EvalSample]:
        """Parse metadata.csv và trả về danh sách EvalSample hợp lệ.

        Args:
            dataset_path: Đường dẫn đến thư mục dataset chứa metadata.csv và wavs/.

        Returns:
            Danh sách EvalSample với sample_id, text, và audio_path.

        Raises:
            FileNotFoundError: Khi dataset_path không tồn tại hoặc thiếu metadata.csv.
        """
        dataset_path = Path(dataset_path)

        # Kiểm tra đường dẫn dataset tồn tại
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {dataset_path}"
            )

        # Kiểm tra metadata.csv tồn tại
        metadata_file = dataset_path / "metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"metadata.csv not found in dataset path: {dataset_path}"
            )

        samples: List[EvalSample] = []

        with open(metadata_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                # Parse pipe-separated format: relative_path|text|duration
                parts = line.split("|")
                if len(parts) < 2:
                    logger.warning(
                        "Skipping invalid line %d in metadata.csv: "
                        "expected at least 2 pipe-separated fields, got %d",
                        line_num,
                        len(parts),
                    )
                    continue

                relative_path = parts[0].strip()
                text = parts[1].strip()

                if not relative_path or not text:
                    logger.warning(
                        "Skipping invalid line %d in metadata.csv: "
                        "empty relative_path or text",
                        line_num,
                    )
                    continue

                # Xây dựng đường dẫn tuyệt đối đến file audio
                audio_path = dataset_path / relative_path

                # Kiểm tra file audio tồn tại trong wavs/
                if not audio_path.exists():
                    logger.warning(
                        "Audio file not found, skipping sample at line %d: %s",
                        line_num,
                        audio_path,
                    )
                    continue

                # Tạo sample_id từ tên file (không có extension)
                sample_id = Path(relative_path).stem

                samples.append(
                    EvalSample(
                        sample_id=sample_id,
                        text=text,
                        audio_path=str(audio_path),
                    )
                )

        logger.info(
            "Loaded %d valid samples from %s", len(samples), metadata_file
        )
        return samples

    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Tải file audio và trả về numpy array cùng sample rate.

        Args:
            path: Đường dẫn đến file audio WAV.

        Returns:
            Tuple gồm:
                - numpy array chứa dữ liệu audio (float32)
                - sample rate (int)

        Raises:
            FileNotFoundError: Khi file audio không tồn tại.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio, sr = librosa.load(path, sr=None)
        return audio, sr
