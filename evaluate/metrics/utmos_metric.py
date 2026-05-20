"""
UTMOS MOS prediction metric.

Sử dụng mô hình UTMOS (sarulab-speech/UTMOS22) từ torch hub
để dự đoán Mean Opinion Score cho audio tổng hợp.
Kết quả nằm trong khoảng [1.0, 5.0].
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Cache model để tránh tải lại nhiều lần
_model: Optional[object] = None


def _load_model() -> object:
    """Tải mô hình UTMOS từ torch hub (có cache).

    Returns:
        Mô hình UTMOS đã tải.
    """
    global _model
    if _model is None:
        logger.info("Đang tải mô hình UTMOS từ torch hub...")
        _model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        _model.eval()
        logger.info("Đã tải mô hình UTMOS thành công")
    return _model


def predict_mos(syn_audio: np.ndarray, sr: int) -> float:
    """Dự đoán Mean Opinion Score cho audio tổng hợp bằng UTMOS.

    Args:
        syn_audio: Audio tổng hợp dạng numpy array (float32).
        sr: Sample rate của audio.

    Returns:
        Điểm MOS dự đoán dạng float trong khoảng [1.0, 5.0].
    """
    model = _load_model()

    # Chuyển audio sang tensor
    audio_tensor = torch.from_numpy(syn_audio.astype(np.float32)).unsqueeze(0)

    # Dự đoán MOS
    with torch.no_grad():
        score = model(audio_tensor, sr)

    # Lấy giá trị scalar
    mos_score = float(score.item()) if hasattr(score, "item") else float(score)

    # Clamp vào khoảng [1.0, 5.0]
    mos_score = float(np.clip(mos_score, 1.0, 5.0))

    logger.debug("UTMOS MOS prediction: %.4f", mos_score)
    return mos_score
