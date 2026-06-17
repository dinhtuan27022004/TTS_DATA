"""
Persistence module: Đọc/ghi metadata synthesis và metric results.

Cấu trúc file:
  Metadata synthesis:
    artifact/results/{dataset}_{ckpt}_metadata.json

  Metric results:
    evaluate/results/{dataset}/{ckpt}/{metric}.json
"""
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def dataset_name_from_path(dataset_path: str) -> str:
    """Lấy tên dataset (basename) từ đường dẫn."""
    return os.path.basename(os.path.normpath(dataset_path))


def _atomic_write(path: str, data: object) -> None:
    """Ghi JSON ra file an toàn (atomic write dùng file tạm và os.replace)."""
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, path)
    except Exception as exc:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise exc


def _safe_read(path: str) -> Optional[object]:
    """Đọc JSON, trả về None nếu lỗi."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Không đọc được %s: %s", path, exc)
        return None


# ─── Synthesis metadata ────────────────────────────────────────────────────────

def synthesis_meta_path(artifact_results_dir: str, dataset_name: str, ckpt_name: str) -> str:
    """Đường dẫn file metadata synthesis."""
    return os.path.join(
        artifact_results_dir, f"{dataset_name}_{ckpt_name}_metadata.json"
    )


def load_synthesis_metadata(meta_path: str) -> List[dict]:
    """Đọc metadata synthesis. Trả về [] nếu chưa có file."""
    data = _safe_read(meta_path)
    if data is None:
        return []
    if not isinstance(data, list):
        logger.warning("Metadata file không phải list: %s", meta_path)
        return []
    return data


def save_synthesis_metadata(meta_path: str, entries: List[dict]) -> None:
    """Ghi toàn bộ metadata synthesis ra file."""
    _atomic_write(meta_path, entries)


# ─── Metric results ────────────────────────────────────────────────────────────

def metric_result_path(
    results_dir: str, dataset_name: str, ckpt_name: str, metric: str
) -> str:
    """Đường dẫn file kết quả metric."""
    return os.path.join(results_dir, dataset_name, ckpt_name, f"{metric}.json")


def load_metric_results(
    results_dir: str, dataset_name: str, ckpt_name: str, metric: str
) -> Dict[str, dict]:
    """Đọc kết quả metric đã lưu.

    Returns:
        Dict wav_file → sample_entry (để tra cứu O(1) khi resume).
    """
    path = metric_result_path(results_dir, dataset_name, ckpt_name, metric)
    data = _safe_read(path)
    if data is None:
        return {}
    samples = data.get("samples", []) if isinstance(data, dict) else []
    return {s["wav_file"]: s for s in samples if "wav_file" in s}


def save_metric_results(
    results_dir: str,
    dataset_name: str,
    ckpt_name: str,
    metric: str,
    samples: List[dict],
) -> None:
    """Lưu kết quả metric vào file JSON.

    Format sample:
      - Audio metrics:  {wav_file, value}
      - WER/CER:        {wav_file, value, asr_transcript}
    """
    path = metric_result_path(results_dir, dataset_name, ckpt_name, metric)

    values = [s["value"] for s in samples if s.get("value") is not None]
    summary: dict = {}
    if values:
        summary = {
            "mean":  float(np.mean(values)),
            "std":   float(np.std(values)),
            "min":   float(min(values)),
            "max":   float(max(values)),
            "count": len(values),
        }

    _atomic_write(path, {
        "dataset":    dataset_name,
        "checkpoint": ckpt_name,
        "metric":     metric,
        "summary":    summary,
        "samples":    samples,
    })
