"""
Word Error Rate (WER) metric.

Sử dụng ASR model nvidia/parakeet-ctc-0.6b-vi (NeMo) để nhận dạng giọng nói
từ audio tổng hợp, sau đó tính WER so với text gốc bằng jiwer.
"""

import logging
import os
import re
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from jiwer import wer, cer

logger = logging.getLogger(__name__)

# Cache model
_asr_model = None

_MODEL_NAME = "
"
_TARGET_SR = 24000


def _load_asr_model():
    """Tải ASR model NeMo (có cache).

    Returns:
        NeMo ASR model đã tải.
    """
    global _asr_model
    if _asr_model is None:
        import nemo.collections.asr as nemo_asr

        logger.info("Đang tải ASR model: %s", _MODEL_NAME)
        _asr_model = nemo_asr.models.ASRModel.from_pretrained(_MODEL_NAME)
        _asr_model.eval()
        logger.info("Đã tải ASR model thành công")
    return _asr_model


def _normalize_text(text: str) -> str:
    """Chuẩn hóa text trước khi tính WER.

    - Chuyển về lowercase
    - Loại bỏ dấu câu
    - Chuẩn hóa khoảng trắng
    - Giữ nguyên dấu tiếng Việt

    Args:
        text: Text cần chuẩn hóa.

    Returns:
        Text đã chuẩn hóa.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _transcribe(audio: np.ndarray, sr: int) -> str:
    """Nhận dạng giọng nói từ audio bằng NeMo ASR model.

    NeMo transcribe nhận file path, nên cần lưu audio tạm ra file WAV.

    Args:
        audio: Audio dạng numpy array (float32).
        sr: Sample rate của audio.

    Returns:
        Text nhận dạng được.
    """
    import librosa

    model = _load_asr_model()

    # Resample về 16000 Hz nếu cần
    if sr != _TARGET_SR:
        audio = librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=_TARGET_SR
        )

    # NeMo transcribe cần file path -> lưu tạm
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio.astype(np.float32), _TARGET_SR)

    try:
        output = model.transcribe([tmp_path])
        # NeMo trả về list of Hypothesis objects hoặc list of strings
        if hasattr(output[0], "text"):
            transcription = output[0].text
        else:
            transcription = str(output[0])
    finally:
        os.unlink(tmp_path)

    return transcription


def compute_wer(syn_audio: np.ndarray, reference_text: str, sr: int) -> dict:
    """Tính Word Error Rate giữa text gốc và text nhận dạng từ audio.

    Sử dụng nvidia/parakeet-ctc-0.6b-vi (NeMo) để nhận dạng giọng nói
    từ audio tổng hợp, chuẩn hóa cả hai text, rồi tính WER bằng jiwer.

    Args:
        syn_audio: Audio tổng hợp dạng numpy array (float32).
        reference_text: Text gốc để so sánh.
        sr: Sample rate của audio.

    Returns:
        Dict với keys: "wer" (float >= 0.0), "transcription" (str nhận dạng được).
    """
    # Nhận dạng giọng nói
    transcription = _transcribe(syn_audio, sr)

    # Chuẩn hóa text
    ref_normalized = _normalize_text(reference_text)
    hyp_normalized = _normalize_text(transcription)

    logger.debug("Reference (normalized): '%s'", ref_normalized)
    logger.debug("Hypothesis (normalized): '%s'", hyp_normalized)

    # Xử lý trường hợp reference rỗng
    if not ref_normalized:
        logger.warning("Reference text rỗng sau khi chuẩn hóa")
        wer_score = 0.0 if not hyp_normalized else 1.0
        cer_score = 0.0 if not hyp_normalized else 1.0
    else:
        wer_score = wer(ref_normalized, hyp_normalized)
        wer_score = float(max(0.0, wer_score))
        cer_score = cer(ref_normalized, hyp_normalized)
        cer_score = float(max(0.0, cer_score))

    logger.debug("WER: %.4f, CER: %.4f", wer_score, cer_score)
    return {"wer": wer_score, "cer": cer_score, "transcription": transcription}
