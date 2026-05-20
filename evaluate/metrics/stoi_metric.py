"""
STOI (Short-Time Objective Intelligibility) metric.

Sử dụng thư viện pystoi để tính điểm STOI giữa audio tham chiếu
và audio tổng hợp. Kết quả nằm trong khoảng [0, 1].
"""

import logging

import numpy as np
from pystoi import stoi

logger = logging.getLogger(__name__)


def compute_stoi(ref_audio: np.ndarray, syn_audio: np.ndarray, sr: int) -> float:
    """Tính điểm STOI giữa audio tham chiếu và audio tổng hợp.

    Căn chỉnh hai tín hiệu về cùng độ dài trước khi tính toán.
    Kết quả nằm trong khoảng [0, 1], giá trị cao hơn cho thấy
    độ rõ ràng tốt hơn.

    Args:
        ref_audio: Audio tham chiếu dạng numpy array (float32).
        syn_audio: Audio tổng hợp dạng numpy array (float32).
        sr: Sample rate của cả hai tín hiệu.

    Returns:
        Điểm STOI dạng float trong khoảng [0, 1].
    """
    # Căn chỉnh độ dài: cắt tín hiệu dài hơn về cùng độ dài ngắn hơn
    min_len = min(len(ref_audio), len(syn_audio))
    ref_audio = ref_audio[:min_len]
    syn_audio = syn_audio[:min_len]

    # Tính STOI
    score = stoi(ref_audio, syn_audio, sr, extended=False)

    # Clamp kết quả vào [0, 1]
    score = float(np.clip(score, 0.0, 1.0))

    logger.debug("STOI score: %.4f", score)
    return score
