"""
MetricCalculator – Tính tất cả metric cho một cặp audio.

Metrics: PESQ, STOI, UTMOS, F0 Correlation, WER, CER.
(MCD đã bị loại bỏ.)
"""
import logging
from typing import Optional

import numpy as np

from evaluate.metrics.pesq_metric import compute_pesq
from evaluate.metrics.stoi_metric import compute_stoi
from evaluate.metrics.utmos_metric import predict_mos
from evaluate.metrics.f0_metric import compute_f0_correlation
from evaluate.metrics.wer_metric import compute_wer
from evaluate.models import MetricResult

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Tính toàn bộ metric đánh giá TTS cho một cặp audio.

    Mỗi metric được bọc try/except riêng – lỗi một metric
    không ảnh hưởng đến các metric còn lại.
    """

    def compute_all(
        self,
        ref_audio: np.ndarray,
        syn_audio: np.ndarray,
        sr: int,
        text: str,
        sample_id: str = "",
    ) -> MetricResult:
        """Tính toàn bộ metric cho cặp (ref_audio, syn_audio).

        Args:
            ref_audio:  Audio tham chiếu (float32 numpy array).
            syn_audio:  Audio tổng hợp (float32 numpy array).
            sr:         Sample rate (cả hai tín hiệu).
            text:       Text gốc (dùng cho WER/CER).
            sample_id:  ID mẫu (dùng cho logging).

        Returns:
            MetricResult với các giá trị metric (None nếu tính lỗi).
        """
        pesq_score:   Optional[float] = None
        stoi_score:   Optional[float] = None
        utmos_score:  Optional[float] = None
        f0_corr:      Optional[float] = None
        wer_score:    Optional[float] = None
        cer_score:    Optional[float] = None
        transcription: Optional[str]  = None

        try:
            pesq_score = compute_pesq(ref_audio, syn_audio, sr)
        except Exception as e:
            logger.error("PESQ [%s]: %s", sample_id, e)

        try:
            stoi_score = compute_stoi(ref_audio, syn_audio, sr)
        except Exception as e:
            logger.error("STOI [%s]: %s", sample_id, e)

        try:
            utmos_score = predict_mos(syn_audio, sr)
        except Exception as e:
            logger.error("UTMOS [%s]: %s", sample_id, e)

        try:
            f0_corr = compute_f0_correlation(ref_audio, syn_audio, sr)
        except Exception as e:
            logger.error("F0 [%s]: %s", sample_id, e)

        try:
            wer_result    = compute_wer(syn_audio, text, sr)
            wer_score     = wer_result["wer"]
            cer_score     = wer_result["cer"]
            transcription = wer_result["transcription"]
        except Exception as e:
            logger.error("WER [%s]: %s", sample_id, e)

        return MetricResult(
            sample_id=sample_id,
            text=text,
            pesq=pesq_score,
            stoi=stoi_score,
            utmos=utmos_score,
            f0_correlation=f0_corr,
            wer=wer_score,
            cer=cer_score,
            transcription=transcription,
        )
