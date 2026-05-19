"""
TTS Evaluation Package
======================

Hệ thống đánh giá chất lượng mô hình Text-to-Speech (TTS) thông qua nhiều metric:
- MOS prediction (UTMOS)
- Mel Cepstral Distortion (MCD)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- F0 Correlation
- Word Error Rate (WER)

Usage:
    from evaluate.models import EvalSample, MetricResult, EvaluationReport
    from evaluate.models import RunHistoryEntry, ResultFileData, TTSModel
"""

from evaluate.models import (
    EvalSample,
    MetricResult,
    EvaluationReport,
    RunHistoryEntry,
    ResultFileData,
    TTSModel,
)

__all__ = [
    # Data models
    "EvalSample",
    "MetricResult",
    "EvaluationReport",
    "RunHistoryEntry",
    "ResultFileData",
    "TTSModel",
]
