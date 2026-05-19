"""
Module persistence cho hệ thống đánh giá TTS.

Cung cấp các hàm lưu/đọc ResultFile và RunHistory:
- save_result_file / load_result_file: Lưu/đọc kết quả metric dạng JSON
- save_run_history / load_run_history: Lưu/đọc lịch sử chạy đánh giá
- sanitize_filename: Chuẩn hóa tên file
- ensure_results_folder: Tạo thư mục kết quả
"""

import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from evaluate.models import RunHistoryEntry

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def sanitize_filename(model_name: str) -> str:
    """Chuẩn hóa tên model thành tên file hợp lệ.

    Thay thế các ký tự đặc biệt không hợp lệ cho tên file bằng dấu gạch dưới.
    Giữ lại: chữ cái, số, dấu gạch ngang, gạch dưới, dấu chấm.

    Args:
        model_name: Tên model gốc.

    Returns:
        Tên file đã chuẩn hóa.
    """
    # Thay thế ký tự không hợp lệ bằng underscore
    sanitized = re.sub(r'[^\w\-.]', '_', model_name)
    # Loại bỏ underscore liên tiếp
    sanitized = re.sub(r'_+', '_', sanitized)
    # Loại bỏ underscore ở đầu/cuối
    sanitized = sanitized.strip('_')
    return sanitized if sanitized else "unknown_model"


def save_result_file(
    model_name: str,
    metric_name: str,
    samples: List[Dict],
    summary: Dict[str, float],
    output_dir: str,
) -> str:
    """Lưu kết quả metric vào file JSON theo quy ước đặt tên.

    File được lưu với tên: <MODEL_NAME>_<METRIC>.json

    Args:
        model_name: Tên mô hình TTS.
        metric_name: Tên metric (mcd, pesq, stoi, utmos, f0_correlation, wer).
        samples: Danh sách kết quả từng mẫu.
            Mỗi sample: {"sample_id": str, "value": float|None, "text": str}
        summary: Thống kê tổng hợp {"mean": float, "std": float, "min": float, "max": float}.
        output_dir: Thư mục lưu file.

    Returns:
        Đường dẫn file JSON đã lưu.
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_name = sanitize_filename(model_name)
    filename = f"{safe_name}_{metric_name}.json"
    file_path = os.path.join(output_dir, filename)

    data = {
        "model_name": model_name,
        "metric_name": metric_name,
        "samples": samples,
        "summary": summary,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"ResultFile saved: {file_path}")
    return file_path


def load_result_file(file_path: str) -> Optional[Dict]:
    """Đọc và parse ResultFile JSON.

    Args:
        file_path: Đường dẫn đến file ResultFile JSON.

    Returns:
        Dict chứa dữ liệu ResultFile, hoặc None nếu file không tồn tại
        hoặc JSON không hợp lệ.
    """
    if not os.path.exists(file_path):
        logger.warning(f"ResultFile not found: {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load ResultFile {file_path}: {e}")
        return None


def save_run_history(entry: RunHistoryEntry, history_path: str) -> None:
    """Append entry vào file RunHistory JSON (không ghi đè dữ liệu cũ).

    Args:
        entry: RunHistoryEntry chứa thông tin lần chạy.
        history_path: Đường dẫn file RunHistory JSON.
    """
    # Tạo thư mục cha nếu chưa tồn tại
    parent_dir = Path(history_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Đọc history hiện có
    history = load_run_history(history_path)

    # Append entry mới
    history.append(asdict(entry))

    # Lưu lại
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info(f"RunHistory updated: {history_path} ({len(history)} entries)")


def load_run_history(history_path: str) -> List[Dict]:
    """Đọc file RunHistory JSON.

    Args:
        history_path: Đường dẫn file RunHistory JSON.

    Returns:
        Danh sách các entry (dict). Trả về list rỗng nếu file không tồn tại.
    """
    if not os.path.exists(history_path):
        return []

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning(f"RunHistory file is not a list: {history_path}")
        return []
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load RunHistory {history_path}: {e}")
        return []


def ensure_results_folder(results_dir: str = DEFAULT_RESULTS_DIR) -> str:
    """Tạo thư mục ResultsFolder nếu chưa tồn tại.

    Args:
        results_dir: Đường dẫn thư mục kết quả.
            Mặc định: evaluate/results/ (relative to package)

    Returns:
        Đường dẫn thư mục đã tạo/tồn tại.
    """
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
