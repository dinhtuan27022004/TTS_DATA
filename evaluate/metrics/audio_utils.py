"""Shared audio preparation helpers for objective TTS metrics."""

from __future__ import annotations

import logging
from typing import Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    """Return a clean mono float32 signal."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if audio.size == 0:
        return audio.astype(np.float32)
    return (audio - float(np.mean(audio))).astype(np.float32)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize peak level without amplifying silence."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < _EPS:
        return audio.astype(np.float32)
    return (audio * min(target_peak / peak, 20.0)).astype(np.float32)


def trim_silence(audio: np.ndarray, top_db: float = 45.0) -> np.ndarray:
    """Trim leading/trailing silence, keeping the original if trim is unsafe."""
    audio = to_mono_float(audio)
    if audio.size == 0 or float(np.max(np.abs(audio))) < _EPS:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    if trimmed.size < max(256, int(0.15 * audio.size)):
        return audio
    return trimmed.astype(np.float32)


def match_rms(ref_audio: np.ndarray, syn_audio: np.ndarray) -> np.ndarray:
    """Scale synthesized audio near reference loudness with conservative bounds."""
    ref_rms = float(np.sqrt(np.mean(np.square(ref_audio)))) if ref_audio.size else 0.0
    syn_rms = float(np.sqrt(np.mean(np.square(syn_audio)))) if syn_audio.size else 0.0
    if ref_rms < _EPS or syn_rms < _EPS:
        return syn_audio.astype(np.float32)
    gain = float(np.clip(ref_rms / syn_rms, 0.25, 4.0))
    return (syn_audio * gain).astype(np.float32)


def fit_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Crop or zero-pad audio to exactly target_len samples."""
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size >= target_len:
        return audio[:target_len].astype(np.float32)
    return np.pad(audio, (0, target_len - audio.size)).astype(np.float32)


def time_align_to_reference(
    ref_audio: np.ndarray,
    syn_audio: np.ndarray,
    max_stretch_ratio: float = 1.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Globally align synthesized duration to reference duration for paired metrics.

    TTS output often has a different speaking rate from the recording. Intrusive
    metrics such as PESQ/STOI assume time alignment, so a conservative global
    time-stretch is less misleading than comparing sample index zero to zero.
    """
    ref_audio = to_mono_float(ref_audio)
    syn_audio = to_mono_float(syn_audio)
    if ref_audio.size == 0 or syn_audio.size == 0:
        return ref_audio, syn_audio

    ratio = syn_audio.size / max(ref_audio.size, 1)
    if 1.0 / max_stretch_ratio <= ratio <= max_stretch_ratio:
        try:
            syn_audio = librosa.effects.time_stretch(syn_audio, rate=ratio)
        except Exception as exc:
            logger.debug("Could not time-stretch synthesized audio: %s", exc)

    target_len = min(ref_audio.size, syn_audio.size)
    ref_audio = fit_length(ref_audio, target_len)
    syn_audio = fit_length(syn_audio, target_len)
    return ref_audio, syn_audio


def prepare_intrusive_pair(
    ref_audio: np.ndarray,
    syn_audio: np.ndarray,
    normalize_peak: bool = True,
    align_duration: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare a reference/synthesis pair for intrusive quality metrics."""
    ref_audio = trim_silence(ref_audio)
    syn_audio = trim_silence(syn_audio)
    if align_duration:
        ref_audio, syn_audio = time_align_to_reference(ref_audio, syn_audio)
    else:
        target_len = min(ref_audio.size, syn_audio.size)
        ref_audio = fit_length(ref_audio, target_len)
        syn_audio = fit_length(syn_audio, target_len)
    syn_audio = match_rms(ref_audio, syn_audio)
    if normalize_peak:
        ref_audio = peak_normalize(ref_audio)
        syn_audio = peak_normalize(syn_audio)
    return ref_audio.astype(np.float32), syn_audio.astype(np.float32)
