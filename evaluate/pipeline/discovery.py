"""
Discovery module: Tìm checkpoints và dataset samples.
"""
import glob
import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Thứ tự ưu tiên khi auto-detect (từ baseline → mới nhất)
_DEFAULT_ORDER = [
    "f5-tts-v0",
    "f5-tts-50000",
    "f5-tts-60000",
    "f5-tts-70000",
]


def get_checkpoints(
    models_dir: str,
    names: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Tìm các checkpoint trong models_dir.

    Args:
        models_dir: Thư mục chứa các folder checkpoint.
        names:      Danh sách tên muốn dùng. None = tự detect theo thứ tự mặc định.

    Returns:
        List of (checkpoint_name, ckpt_path) theo thứ tự ưu tiên.
    """
    if names is None:
        names = _DEFAULT_ORDER

    result: List[Tuple[str, str]] = []
    for name in names:
        folder = os.path.join(models_dir, name)
        if not os.path.isdir(folder):
            continue
        pt_files = sorted(f for f in os.listdir(folder) if f.endswith(".pt"))
        if not pt_files:
            logger.debug("Không có .pt trong %s – bỏ qua", folder)
            continue
        ckpt_path = os.path.join(folder, pt_files[0])
        result.append((name, ckpt_path))

    logger.info(
        "Checkpoints được dùng (%d): %s",
        len(result),
        [n for n, _ in result],
    )
    return result


def get_dataset_samples(dataset_path: str) -> List[dict]:
    """Quét folder dataset, trả về danh sách sample dicts.

    Mỗi sample: {sample_id, wav_path, txt_path, text}
    Yêu cầu: mỗi file .wav phải có file .txt cùng tên.

    Args:
        dataset_path: Thư mục chứa các cặp file .wav + .txt.

    Returns:
        List of sample dicts, sorted by sample_id.
    """
    wav_files = sorted(
        glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    )

    samples: List[dict] = []
    skipped = 0

    for wav_path in wav_files:
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.isfile(txt_path):
            skipped += 1
            continue
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                skipped += 1
                continue
            sample_id = os.path.splitext(os.path.basename(wav_path))[0]
            samples.append(
                {
                    "sample_id": sample_id,
                    "wav_path":  wav_path,
                    "txt_path":  txt_path,
                    "text":      text,
                }
            )
        except Exception as exc:
            logger.warning("Lỗi đọc %s: %s", txt_path, exc)
            skipped += 1

    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    logger.info(
        "Dataset '%s': %d samples (%d bỏ qua do thiếu .txt hoặc rỗng)",
        dataset_name, len(samples), skipped,
    )
    return samples
