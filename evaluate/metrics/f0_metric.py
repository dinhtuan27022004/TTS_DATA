"""
F0 Correlation metric.

Trích xuất F0 contour từ audio tham chiếu và audio tổng hợp
bằng librosa.pyin, sau đó tính hệ số tương quan Pearson.
"""

import logging
from typing import Optional

import librosa
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def compute_f0_correlation(
    ref_audio: np.ndarray, syn_audio: np.ndarray, sr: int
) -> Optional[float]:
    """Tính hệ số tương quan Pearson giữa F0 contour của hai tín hiệu.

    Sử dụng librosa.pyin để trích xuất F0. Chỉ tính tương quan
    trên các frame mà cả hai tín hiệu đều có voiced (F0 > 0).

    Args:
        ref_audio: Audio tham chiếu dạng numpy array (float32).
        syn_audio: Audio tổng hợp dạng numpy array (float32).
        sr: Sample rate của cả hai tín hiệu.

    Returns:
        Hệ số tương quan Pearson trong khoảng [-1.0, 1.0],
        hoặc None nếu không thể trích xuất F0.
    """
    # Thiết lập khoảng tần số F0 cho giọng nói
    fmin = librosa.note_to_hz("C2")  # ~65 Hz
    fmax = librosa.note_to_hz("C7")  # ~2093 Hz

    # Trích xuất F0 contour bằng pyin
    f0_ref, voiced_ref, _ = librosa.pyin(
        ref_audio.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr
    )
    f0_syn, voiced_syn, _ = librosa.pyin(
        syn_audio.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr
    )

    if f0_ref is None or f0_syn is None:
        logger.warning("Không thể trích xuất F0 từ một trong hai tín hiệu")
        return None

    # Căn chỉnh độ dài F0 contour
    min_len = min(len(f0_ref), len(f0_syn))
    f0_ref = f0_ref[:min_len]
    f0_syn = f0_syn[:min_len]
    voiced_ref = voiced_ref[:min_len]
    voiced_syn = voiced_syn[:min_len]

    # Chỉ lấy các frame mà cả hai đều voiced
    both_voiced = voiced_ref & voiced_syn

    if not np.any(both_voiced):
        logger.warning(
            "Không có frame nào cả hai tín hiệu đều voiced, "
            "không thể tính F0 correlation"
        )
        return None

    f0_ref_voiced = f0_ref[both_voiced]
    f0_syn_voiced = f0_syn[both_voiced]

    # Cần ít nhất 2 điểm để tính tương quan
    if len(f0_ref_voiced) < 2:
        logger.warning("Không đủ voiced frames để tính F0 correlation")
        return None

    # Tính hệ số tương quan Pearson
    correlation, _ = stats.pearsonr(f0_ref_voiced, f0_syn_voiced)

    # Xử lý trường hợp NaN (khi std = 0)
    if np.isnan(correlation):
        logger.warning("F0 correlation là NaN (có thể do std = 0)")
        return None

    # Clamp vào [-1.0, 1.0] (phòng trường hợp lỗi số học)
    correlation = float(np.clip(correlation, -1.0, 1.0))

    logger.debug("F0 correlation: %.4f", correlation)
    return correlation
