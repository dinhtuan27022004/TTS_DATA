"""
F0 Correlation metric.

Trich xuat F0 contour tu audio tham chieu va audio tong hop,
can chinh contour bang DTW, sau do tinh he so tuong quan Pearson
tren log-F0. Cach nay on dinh hon cho TTS vi toc do noi cua audio
sinh ra thuong khac ban ghi tham chieu.
"""

import logging
from typing import Optional, Tuple

import librosa
import numpy as np
from scipy import stats

from evaluate.metrics.audio_utils import trim_silence, to_mono_float

logger = logging.getLogger(__name__)

_FMIN_HZ = 50.0
_FMAX_HZ = 650.0
_FRAME_LENGTH = 2048
_HOP_LENGTH = 256
_MIN_VOICED_FRAMES = 5


def _extract_log_f0(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract voiced log2-F0 values from a speech signal."""
    audio = trim_silence(to_mono_float(audio))
    if audio.size < _FRAME_LENGTH:
        return np.zeros(0, dtype=np.float32)

    f0, voiced, _ = librosa.pyin(
        audio.astype(np.float32),
        fmin=_FMIN_HZ,
        fmax=_FMAX_HZ,
        sr=sr,
        frame_length=_FRAME_LENGTH,
        hop_length=_HOP_LENGTH,
    )
    if f0 is None or voiced is None:
        return np.zeros(0, dtype=np.float32)

    valid = voiced & np.isfinite(f0) & (f0 > 0.0)
    if not np.any(valid):
        return np.zeros(0, dtype=np.float32)

    log_f0 = np.log2(f0[valid].astype(np.float64))
    median = float(np.median(log_f0))
    mad = float(np.median(np.abs(log_f0 - median)))
    if mad > 0.0:
        keep = np.abs(log_f0 - median) <= 6.0 * mad
        log_f0 = log_f0[keep]
    return log_f0.astype(np.float32)


def _dtw_pair_contours(ref_log_f0: np.ndarray, syn_log_f0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pair two log-F0 contours with DTW."""
    _, path = librosa.sequence.dtw(
        X=ref_log_f0[np.newaxis, :],
        Y=syn_log_f0[np.newaxis, :],
        metric="euclidean",
    )
    path = path[::-1]
    return ref_log_f0[path[:, 0]], syn_log_f0[path[:, 1]]


def compute_f0_correlation(
    ref_audio: np.ndarray, syn_audio: np.ndarray, sr: int
) -> Optional[float]:
    """Tinh Pearson correlation giua hai F0 contour da can chinh.

    Returns:
        He so tuong quan trong khoang [-1.0, 1.0], hoac None neu khong du
        voiced frames de danh gia pitch/intonation.
    """
    ref_log_f0 = _extract_log_f0(ref_audio, sr)
    syn_log_f0 = _extract_log_f0(syn_audio, sr)

    if len(ref_log_f0) < _MIN_VOICED_FRAMES or len(syn_log_f0) < _MIN_VOICED_FRAMES:
        logger.warning(
            "Khong du voiced frames de tinh F0 correlation (ref=%d, syn=%d)",
            len(ref_log_f0),
            len(syn_log_f0),
        )
        return None

    try:
        paired_ref, paired_syn = _dtw_pair_contours(ref_log_f0, syn_log_f0)
    except Exception as exc:
        logger.warning("Khong the DTW F0 contour: %s", exc)
        min_len = min(len(ref_log_f0), len(syn_log_f0))
        paired_ref = ref_log_f0[:min_len]
        paired_syn = syn_log_f0[:min_len]

    if len(paired_ref) < _MIN_VOICED_FRAMES:
        logger.warning("Khong du F0 frames sau khi can chinh")
        return None

    ref_std = float(np.std(paired_ref))
    syn_std = float(np.std(paired_syn))
    if ref_std < 1e-5 or syn_std < 1e-5:
        cents_rmse = float(np.sqrt(np.mean(np.square(paired_ref - paired_syn))) * 1200.0)
        score = 1.0 - min(cents_rmse / 300.0, 2.0)
        return float(np.clip(score, -1.0, 1.0))

    correlation, _ = stats.pearsonr(paired_ref, paired_syn)
    if np.isnan(correlation):
        logger.warning("F0 correlation la NaN")
        return None

    correlation = float(np.clip(correlation, -1.0, 1.0))
    logger.debug("F0 correlation: %.4f", correlation)
    return correlation
