"""
PESQ (Perceptual Evaluation of Speech Quality) metric.

Sử dụng thư viện pesq để tính điểm PESQ giữa audio tham chiếu
và audio tổng hợp. Hỗ trợ resample tự động nếu sample rate
không phải 8000 hoặc 16000 Hz.
"""

import logging
from typing import Optional

import librosa
import numpy as np
from pesq import pesq

logger = logging.getLogger(__name__)

# Sample rate được hỗ trợ bởi PESQ
_SUPPORTED_RATES = (8000, 16000)
_DEFAULT_RATE = 16000
# Độ dài tối thiểu (samples) để PESQ có thể tính toán
_MIN_SAMPLES = 1024


def compute_pesq(
    ref_audio: np.ndarray, syn_audio: np.ndarray, sr: int
) -> Optional[float]:
    """Tính điểm PESQ giữa audio tham chiếu và audio tổng hợp.

    Tự động resample về 16000 Hz nếu sample rate đầu vào không phải
    8000 hoặc 16000 Hz.

    Args:
        ref_audio: Audio tham chiếu dạng numpy array (float32).
        syn_audio: Audio tổng hợp dạng numpy array (float32).
        sr: Sample rate của cả hai tín hiệu.

    Returns:
        Điểm PESQ dạng float, hoặc None nếu audio quá ngắn.
    """
    target_sr = sr

    # Resample nếu sample rate không được hỗ trợ
    if sr not in _SUPPORTED_RATES:
        target_sr = _DEFAULT_RATE
        ref_audio = librosa.resample(ref_audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        syn_audio = librosa.resample(syn_audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        logger.debug("Resampled audio từ %d Hz về %d Hz cho PESQ", sr, target_sr)

    # Kiểm tra độ dài tối thiểu
    if len(ref_audio) < _MIN_SAMPLES or len(syn_audio) < _MIN_SAMPLES:
        logger.warning(
            "Audio quá ngắn để tính PESQ (ref: %d, syn: %d samples, min: %d)",
            len(ref_audio),
            len(syn_audio),
            _MIN_SAMPLES,
        )
        return None

    # Chọn mode dựa trên sample rate
    mode = "wb" if target_sr == 16000 else "nb"

    try:
        score = pesq(target_sr, ref_audio, syn_audio, mode)
        logger.debug("PESQ score: %.4f (mode=%s)", score, mode)
        return float(score)
    except Exception as e:
        logger.warning("Không thể tính PESQ: %s", str(e))
        return None
