"""
MetricCalculator - Tính toán tất cả metric cho một cặp audio.

Class MetricCalculator cung cấp phương thức compute_all để tính toán
tất cả metric (MCD, PESQ, STOI, UTMOS, F0 Correlation, WER) trong
một lần gọi. Mỗi metric được xử lý độc lập — nếu một metric gặp lỗi,
các metric còn lại vẫn được tính toán bình thường.
"""

import logging
from typing import Optional

import numpy as np

from evaluate.metrics.mcd import compute_mcd
from evaluate.metrics.pesq_metric import compute_pesq
from evaluate.metrics.stoi_metric import compute_stoi
from evaluate.metrics.utmos_metric import predict_mos
from evaluate.metrics.f0_metric import compute_f0_correlation
from evaluate.metrics.wer_metric import compute_wer
from evaluate.models import MetricResult

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Tính toán tất cả metric đánh giá TTS cho một cặp audio.

    Class này đóng vai trò orchestrator, gọi từng metric riêng lẻ
    và tổng hợp kết quả vào MetricResult. Mỗi metric được bọc trong
    try/except để đảm bảo fault tolerance — lỗi ở một metric không
    ảnh hưởng đến các metric khác.
    """

    def compute_all(
        self,
        ref_audio: np.ndarray,
        syn_audio: np.ndarray,
        sr: int,
        text: str,
        sample_id: str = "",
    ) -> MetricResult:
        """Tính toán tất cả metric cho một cặp audio tham chiếu và tổng hợp.

        Args:
            ref_audio: Audio tham chiếu dạng numpy array (float32).
            syn_audio: Audio tổng hợp dạng numpy array (float32).
            sr: Sample rate của cả hai tín hiệu.
            text: Nội dung text gốc (dùng cho WER).
            sample_id: ID của mẫu đánh giá.

        Returns:
            MetricResult chứa kết quả tất cả metric. Metric nào gặp lỗi
            sẽ có giá trị None.
        """
        mcd_score: Optional[float] = None
        pesq_score: Optional[float] = None
        stoi_score: Optional[float] = None
        utmos_score: Optional[float] = None
        f0_corr: Optional[float] = None
        wer_score: Optional[float] = None

        # MCD
        try:
            mcd_score = compute_mcd(ref_audio, syn_audio, sr)
            logger.debug("MCD computed successfully: %.4f", mcd_score)
        except Exception as e:
            logger.error("Lỗi khi tính MCD: %s", str(e))

        # PESQ
        try:
            pesq_score = compute_pesq(ref_audio, syn_audio, sr)
            logger.debug("PESQ computed successfully: %s", pesq_score)
        except Exception as e:
            logger.error("Lỗi khi tính PESQ: %s", str(e))

        # STOI
        try:
            stoi_score = compute_stoi(ref_audio, syn_audio, sr)
            logger.debug("STOI computed successfully: %.4f", stoi_score)
        except Exception as e:
            logger.error("Lỗi khi tính STOI: %s", str(e))

        # UTMOS
        try:
            utmos_score = predict_mos(syn_audio, sr)
            logger.debug("UTMOS computed successfully: %.4f", utmos_score)
        except Exception as e:
            logger.error("Lỗi khi tính UTMOS: %s", str(e))

        # F0 Correlation
        try:
            f0_corr = compute_f0_correlation(ref_audio, syn_audio, sr)
            logger.debug("F0 correlation computed successfully: %s", f0_corr)
        except Exception as e:
            logger.error("Lỗi khi tính F0 correlation: %s", str(e))

        # WER
        transcription: Optional[str] = None
        cer_score: Optional[float] = None
        try:
            wer_result = compute_wer(syn_audio, text, sr)
            wer_score = wer_result["wer"]
            cer_score = wer_result["cer"]
            transcription = wer_result["transcription"]
            logger.debug("WER computed successfully: %s", wer_score)
        except Exception as e:
            logger.error("Lỗi khi tính WER: %s", str(e))

        return MetricResult(
            sample_id=sample_id,
            text=text,
            mcd=mcd_score,
            pesq=pesq_score,
            stoi=stoi_score,
            utmos=utmos_score,
            f0_correlation=f0_corr,
            wer=wer_score,
            cer=cer_score,
            transcription=transcription,
        )
