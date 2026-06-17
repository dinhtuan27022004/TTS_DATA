"""
Word Error Rate (WER) / CER metric.

ASR backend: openai/whisper-large-v3 (thay thế NeMo parakeet-ctc-0.6b-vi).
Model được cache toàn bộ phiên (singleton), chỉ load 1 lần.
"""
import logging
import os
import re
import tempfile

import numpy as np
import soundfile as sf
from jiwer import wer, cer
from components.asr import get_whisper_worker

logger = logging.getLogger(__name__)

_TARGET_SR    = 16_000


def _normalize_text(text: str) -> str:
    """Chuẩn hóa text trước khi tính WER (lowercase, bỏ dấu câu, trim)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _transcribe(audio: np.ndarray, sr: int) -> str:
    """Nhận dạng giọng nói từ numpy array bằng Whisper."""
    import librosa

    # Resample về 16 kHz (Whisper yêu cầu)
    if sr != _TARGET_SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=_TARGET_SR)

    # Whisper nhận file path → lưu tạm
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio.astype(np.float32), _TARGET_SR)

    try:
        worker = get_whisper_worker(
            model_name="large-v3",
            language="vi",
            device="cuda",
            compute_type="float16",
            beam_size=5,
            num_workers=1,
            word_timestamps=False,
        )
        return worker.transcribe_text(tmp_path)
    finally:
        os.unlink(tmp_path)


def compute_wer(syn_audio: np.ndarray, reference_text: str, sr: int) -> dict:
    """Tính WER và CER giữa text gốc và kết quả ASR (Whisper Large v3).

    Args:
        syn_audio:      Audio tổng hợp dạng numpy array (float32).
        reference_text: Text gốc để so sánh.
        sr:             Sample rate của audio.

    Returns:
        Dict với keys: wer (float), cer (float), transcription (str).
    """
    transcription = _transcribe(syn_audio, sr)

    ref_n = _normalize_text(reference_text)
    hyp_n = _normalize_text(transcription)

    if not ref_n:
        return {
            "wer": 0.0 if not hyp_n else 1.0,
            "cer": 0.0,
            "transcription": transcription,
        }

    wer_score = float(max(0.0, wer(ref_n, hyp_n)))
    cer_score = float(max(0.0, cer(ref_n, hyp_n)))

    logger.debug("WER=%.4f  CER=%.4f", wer_score, cer_score)
    return {"wer": wer_score, "cer": cer_score, "transcription": transcription}
