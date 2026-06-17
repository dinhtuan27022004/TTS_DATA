"""
Phase 2 – Metric Computation.

Chạy sau khi synthesis xong. Tính các metric cho từng cặp (ref_wav, syn_wav):
  - UTMOS, F0 Correlation, Speaker Similarity  ← so sánh audio ref vs syn
  - WER, CER                            ← ASR via Whisper Large v3 trên syn

Resume: bỏ qua sample đã tính (dựa vào JSON đã lưu trong evaluate/results/).
"""
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np

from evaluate.pipeline.persistence import (
    load_metric_results,
    load_synthesis_metadata,
    save_metric_results,
)
from components.asr import get_whisper_worker

logger = logging.getLogger(__name__)

# faster-whisper/CTranslate2 logs one line per file by default; keep pipeline logs readable.
for _noisy_logger in (
    "faster_whisper",
    "faster_whisper.transcribe",
    "ctranslate2",
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# Metric được tính ở Phase 2
AUDIO_METRICS = ["utmos", "f0_corr", "speaker_sim"]  # cần ref + syn audio
# Ưu tiên metric thường nhẹ trước để có chart sớm; UTMOS/SpeakerSim để sau vì load model neural.
AUDIO_METRIC_ORDER = ["f0_corr", "utmos", "speaker_sim"]
ASR_METRICS   = ["wer", "cer"]                          # cần Whisper
ALL_METRICS   = AUDIO_METRICS + ASR_METRICS

TARGET_SR = 24_000  # resample về 24 kHz trước khi tính metric
METRIC_WORKERS = 4
WHISPER_NUM_WORKERS = 4
WHISPER_BATCH_SIZE = 16
LOG_EVERY_SAMPLES = 100
CHART_EVERY_SAMPLES = 100


def _transcribe(wav_path: str, batch_size: int = WHISPER_BATCH_SIZE) -> str:
    """Nhận dạng tiếng Việt từ file WAV bằng Whisper."""
    worker = get_whisper_worker(
        model_name="large-v3",
        language="vi",
        device="cuda",
        compute_type="float16",
        beam_size=5,
        num_workers=WHISPER_NUM_WORKERS,
        word_timestamps=False,
        use_batched_pipeline=True,
    )
    return worker.transcribe_text(wav_path, batch_size=batch_size)


# ─── Audio helpers ────────────────────────────────────────────────────────────

def _load_audio_pair(
    ref_path: str, syn_path: str, target_sr: int = TARGET_SR
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Load và resample cặp (ref_wav, syn_wav). Trả về None nếu lỗi."""
    try:
        ref, ref_sr = librosa.load(ref_path, sr=None)
        syn, syn_sr = librosa.load(syn_path, sr=None)
    except Exception as exc:
        logger.error("Không load được audio: %s", exc)
        return None

    if ref_sr != target_sr:
        ref = librosa.resample(ref, orig_sr=ref_sr, target_sr=target_sr)
    if syn_sr != target_sr:
        syn = librosa.resample(syn, orig_sr=syn_sr, target_sr=target_sr)

    return ref.astype(np.float32), syn.astype(np.float32), target_sr


# ─── Individual metric computation ───────────────────────────────────────────

def _compute_audio_metrics(
    ref: np.ndarray, syn: np.ndarray, sr: int
) -> Dict[str, Optional[float]]:
    """Tính UTMOS, F0 Corr, Speaker Similarity. Lỗi từng metric không ảnh hưởng metric khác."""
    from evaluate.metrics.utmos_metric import predict_mos
    from evaluate.metrics.f0_metric import compute_f0_correlation
    from evaluate.metrics.speaker_sim_metric import compute_speaker_similarity

    results: Dict[str, Optional[float]] = {}

    for name, fn in [
        ("utmos",       lambda: predict_mos(syn, sr)),
        ("f0_corr",     lambda: compute_f0_correlation(ref, syn, sr)),
        ("speaker_sim", lambda: compute_speaker_similarity(ref, syn, sr)),
    ]:
        try:
            results[name] = fn()
        except Exception as exc:
            logger.warning("Lỗi %s: %s", name.upper(), exc)
            results[name] = None

    return results


def _compute_asr_metrics(syn_wav_path: str, ref_text: str) -> Dict:
    """Tính WER và CER bằng Whisper ASR."""
    try:
        transcript = _transcribe(syn_wav_path)
    except Exception as exc:
        logger.warning("Whisper lỗi: %s", exc)
        return {"wer": None, "cer": None, "asr_transcript": ""}

    return _score_asr_transcript(transcript, ref_text)


def _score_asr_transcript(transcript: str, ref_text: str) -> Dict:
    """Tính WER/CER từ transcript đã có sẵn."""
    from jiwer import wer, cer

    def normalize(t: str) -> str:
        t = t.lower()
        t = re.sub(r"[^\w\s]", "", t)
        return re.sub(r"\s+", " ", t).strip()

    ref_n = normalize(ref_text)
    hyp_n = normalize(transcript)

    if not ref_n:
        return {"wer": 0.0, "cer": 0.0, "asr_transcript": transcript}

    return {
        "wer":            float(max(0.0, wer(ref_n, hyp_n))),
        "cer":            float(max(0.0, cer(ref_n, hyp_n))),
        "asr_transcript": transcript,
    }


def _compute_audio_metrics_for_entry(entry: dict) -> Tuple[str, Optional[Dict[str, Optional[float]]]]:
    """Load audio và tính toàn bộ audio metrics cho một sample."""
    wav_file = entry["wav_file"]
    syn_path = entry["output_path"]
    ref_path = entry["ref_audio"]

    pair = _load_audio_pair(ref_path, syn_path)
    if pair is None:
        return wav_file, None
    ref, syn, sr = pair

    audio_vals = _compute_audio_metrics(ref, syn, sr)
    return wav_file, audio_vals


def _compute_single_audio_metric(metric: str, ref: np.ndarray, syn: np.ndarray, sr: int) -> Optional[float]:
    """Tính một audio metric; import lazy để metric nhẹ không phải load model nặng."""
    if metric == "utmos":
        from evaluate.metrics.utmos_metric import predict_mos
        return predict_mos(syn, sr)
    if metric == "f0_corr":
        from evaluate.metrics.f0_metric import compute_f0_correlation
        return compute_f0_correlation(ref, syn, sr)
    if metric == "speaker_sim":
        from evaluate.metrics.speaker_sim_metric import compute_speaker_similarity
        return compute_speaker_similarity(ref, syn, sr)
    raise ValueError(f"Không hỗ trợ audio metric: {metric}")


def _preload_audio_metric_model(metric: str) -> None:
    """Pre-load model chỉ cho metric đang chạy, tránh chờ model nặng từ đầu."""
    if metric == "utmos":
        from evaluate.metrics.utmos_metric import _load_model
        _load_model()
    elif metric == "speaker_sim":
        from evaluate.metrics.speaker_sim_metric import _load_classifier
        _load_classifier()


def _compute_single_audio_metric_for_entry(entry: dict, metric: str) -> Tuple[str, Optional[float]]:
    """Load audio và tính đúng một metric cho một sample."""
    wav_file = entry["wav_file"]
    syn_path = entry["output_path"]
    ref_path = entry["ref_audio"]

    pair = _load_audio_pair(ref_path, syn_path)
    if pair is None:
        return wav_file, None
    ref, syn, sr = pair

    try:
        return wav_file, _compute_single_audio_metric(metric, ref, syn, sr)
    except Exception as exc:
        logger.warning("Lỗi %s cho %s: %s", metric.upper(), wav_file, exc)
        return wav_file, None


def _compute_asr_batch(entries: List[dict]) -> Dict[str, Dict]:
    """Chạy Whisper theo lô; mỗi file dùng batching nội bộ của faster-whisper."""
    results = {}
    for entry in entries:
        wav_file = entry["wav_file"]
        try:
            transcript = _transcribe(entry["output_path"], batch_size=WHISPER_BATCH_SIZE)
            results[wav_file] = _score_asr_transcript(transcript, entry["gen_text"])
        except Exception as exc:
            logger.warning("Whisper lỗi với %s: %s", wav_file, exc)
            results[wav_file] = {"wer": None, "cer": None, "asr_transcript": ""}
    return results


# ─── Phase 2 entry points ─────────────────────────────────────────────────────

def _load_ok_samples(metadata_results_dir: str, dataset_name: str, ckpt_name: str) -> List[dict]:
    """Đọc metadata synthesis và trả về các sample đã synthesize thành công."""
    meta_path = os.path.join(
        metadata_results_dir,
        f"{dataset_name}_{ckpt_name}_metadata.json",
    )
    metadata = load_synthesis_metadata(meta_path)
    return [e for e in metadata if e.get("status") == "ok"]


def _load_metric_state(
    results_dir: str,
    dataset_name: str,
    ckpt_name: str,
    metrics: List[str],
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, List[dict]]]:
    """Load kết quả metric hiện có để resume."""
    existing: Dict[str, Dict[str, dict]] = {
        m: load_metric_results(results_dir, dataset_name, ckpt_name, m)
        for m in metrics
    }
    metric_samples: Dict[str, List[dict]] = {
        m: list(existing[m].values()) for m in metrics
    }
    return existing, metric_samples


def _missing_metric_samples(
    samples: List[dict],
    existing: Dict[str, Dict[str, dict]],
    metrics: List[str],
) -> List[dict]:
    """Lọc sample chưa đủ các metric yêu cầu."""
    missing = []
    for entry in samples:
        wav_file = entry["wav_file"]
        if not all(wav_file in existing[m] for m in metrics):
            missing.append(entry)
    return missing


def _save_metric_group(
    results_dir: str,
    dataset_name: str,
    ckpt_name: str,
    metrics: List[str],
    metric_samples: Dict[str, List[dict]],
) -> None:
    for metric in metrics:
        save_metric_results(
            results_dir, dataset_name, ckpt_name, metric, metric_samples[metric]
        )


def _run_balanced_audio_metric(
    metric: str,
    checkpoints: List[Tuple[str, str]],
    results_dir: str,
    dataset_name: str,
    metadata_results_dir: str,
    progress_callback: Optional[Callable[[List[str]], None]] = None,
) -> None:
    """Tính một audio metric theo batch liên-checkpoint, ưu tiên checkpoint ít sample nhất."""
    states = {}
    for ckpt_name, _ in checkpoints:
        ok_samples = _load_ok_samples(metadata_results_dir, dataset_name, ckpt_name)
        if not ok_samples:
            logger.warning("[%s] Không có sample synthesis ok. Bỏ qua.", ckpt_name)
            continue

        existing, metric_samples = _load_metric_state(
            results_dir, dataset_name, ckpt_name, [metric]
        )
        to_process = _missing_metric_samples(ok_samples, existing, [metric])
        completed = len(existing[metric])

        states[ckpt_name] = {
            "ok_total": len(ok_samples),
            "existing": existing,
            "metric_samples": metric_samples,
            "pending": to_process,
            "cursor": 0,
            "initial_completed": completed,
            "completed": completed,
            "next_chart_at": ((completed // CHART_EVERY_SAMPLES) + 1) * CHART_EVERY_SAMPLES,
            "new_processed": 0,
        }

        if to_process:
            logger.info(
                "[%s] %s resume: đã có %d/%d, còn %d sample.",
                ckpt_name,
                metric,
                completed,
                len(ok_samples),
                len(to_process),
            )
        else:
            logger.info("[%s] Metric %s đã có đủ %d samples (resume).", ckpt_name, metric, len(ok_samples))

    active = {ckpt for ckpt, state in states.items() if state["pending"]}
    if not active:
        logger.info("Metric %s đã hoàn tất cho mọi checkpoint.", metric)
        return

    while active:
        batch_items = []
        while len(batch_items) < METRIC_WORKERS and active:
            ckpt_name = min(
                active,
                key=lambda name: (states[name]["completed"], name),
            )
            state = states[ckpt_name]
            idx = state["cursor"]
            pending = state["pending"]

            if idx >= len(pending):
                active.remove(ckpt_name)
                continue

            entry = pending[idx]
            state["cursor"] += 1
            state["completed"] += 1
            batch_items.append((ckpt_name, entry))

            if state["cursor"] >= len(pending):
                active.remove(ckpt_name)

        if not batch_items:
            break

        with ThreadPoolExecutor(max_workers=len(batch_items)) as executor:
            futures = {
                executor.submit(_compute_single_audio_metric_for_entry, entry, metric): (ckpt_name, entry)
                for ckpt_name, entry in batch_items
            }

            touched = set()
            last_value_by_ckpt = {}
            for future in as_completed(futures):
                ckpt_name, entry = futures[future]
                wav_file = entry["wav_file"]
                state = states[ckpt_name]
                try:
                    _, value = future.result()
                except Exception as exc:
                    logger.error("Lỗi khi tính %s cho %s: %s", metric, wav_file, exc)
                    value = None

                rec = {"wav_file": wav_file, "value": value}
                state["metric_samples"][metric].append(rec)
                state["existing"][metric][wav_file] = rec
                state["new_processed"] += 1
                touched.add(ckpt_name)
                last_value_by_ckpt[ckpt_name] = value

        for ckpt_name in sorted(touched):
            state = states[ckpt_name]
            _save_metric_group(
                results_dir,
                dataset_name,
                ckpt_name,
                [metric],
                state["metric_samples"],
            )
            if state["completed"] >= state["next_chart_at"] or state["completed"] == state["ok_total"]:
                if progress_callback is not None:
                    progress_callback([metric])
                state["next_chart_at"] = ((state["completed"] // CHART_EVERY_SAMPLES) + 1) * CHART_EVERY_SAMPLES

            if state["completed"] % LOG_EVERY_SAMPLES == 0 or state["completed"] == state["ok_total"]:
                logger.info(
                    "[%s] %s progress: %d/%d  value=%s",
                    ckpt_name,
                    metric.upper(),
                    state["completed"],
                    state["ok_total"],
                    f"{last_value_by_ckpt[ckpt_name]:.3f}" if last_value_by_ckpt[ckpt_name] is not None else "N/A",
                )

    for ckpt_name, state in sorted(states.items()):
        logger.info(
            "[%s] Metric %s xong (%d mới, %d resume).",
            ckpt_name,
            metric,
            state["new_processed"],
            state["initial_completed"],
        )


def run_audio_metric_phase(
    checkpoints: List[Tuple[str, str]],
    results_dir: str,
    dataset_name: str,
    metadata_results_dir: str,
    metrics: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[List[str]], None]] = None,
) -> None:
    """Tính audio metrics từng metric một để metric nào xong có thể vẽ chart ngay."""
    metrics_to_run = metrics or AUDIO_METRIC_ORDER

    for metric in metrics_to_run:
        if metric not in AUDIO_METRICS:
            raise ValueError(f"Không hỗ trợ audio metric: {metric}")

        logger.info("=" * 55)
        logger.info("Audio metric [%s]", metric)
        _preload_audio_metric_model(metric)

        if metric in {"f0_corr", "utmos", "speaker_sim"}:
            _run_balanced_audio_metric(
                metric, checkpoints, results_dir, dataset_name, metadata_results_dir, progress_callback
            )
            continue

        for ckpt_name, _ in checkpoints:
            logger.info("─" * 55)
            logger.info("Audio metric [%s]  checkpoint [%s]", metric, ckpt_name)

            ok_samples = _load_ok_samples(metadata_results_dir, dataset_name, ckpt_name)
            if not ok_samples:
                logger.warning("[%s] Không có sample synthesis ok. Bỏ qua.", ckpt_name)
                continue

            existing, metric_samples = _load_metric_state(
                results_dir, dataset_name, ckpt_name, [metric]
            )
            to_process = _missing_metric_samples(ok_samples, existing, [metric])

            if not to_process:
                logger.info(
                    "[%s] Metric %s đã có đủ %d samples (resume).",
                    ckpt_name,
                    metric,
                    len(ok_samples),
                )
                continue

            logger.info(
                "[%s] Bắt đầu tính %s cho %d/%d samples...",
                ckpt_name,
                metric,
                len(to_process),
                len(ok_samples),
            )

            processed = 0
            last_value: Optional[float] = None
            with ThreadPoolExecutor(max_workers=METRIC_WORKERS) as executor:
                futures = {
                    executor.submit(_compute_single_audio_metric_for_entry, entry, metric): entry
                    for entry in to_process
                }

                for future in as_completed(futures):
                    entry = futures[future]
                    wav_file = entry["wav_file"]
                    try:
                        _, value = future.result()
                    except Exception as exc:
                        logger.error("Lỗi khi tính %s cho %s: %s", metric, wav_file, exc)
                        continue

                    rec = {"wav_file": wav_file, "value": value}
                    metric_samples[metric].append(rec)
                    existing[metric][wav_file] = rec

                    processed += 1
                    last_value = value
                    if processed % CHART_EVERY_SAMPLES == 0 or processed == len(to_process):
                        _save_metric_group(results_dir, dataset_name, ckpt_name, [metric], metric_samples)
                        if progress_callback is not None:
                            progress_callback([metric])

                    if processed % LOG_EVERY_SAMPLES == 0 or processed == len(to_process):
                        logger.info(
                            "[%s] %s progress: %d/%d  value=%s",
                            ckpt_name,
                            metric.upper(),
                            processed,
                            len(to_process),
                            f"{last_value:.3f}" if last_value is not None else "N/A",
                        )

            _save_metric_group(results_dir, dataset_name, ckpt_name, [metric], metric_samples)
            logger.info(
                "[%s] Metric %s xong (%d mới, %d resume).",
                ckpt_name,
                metric,
                processed,
                len(ok_samples) - processed,
            )


def run_asr_metric_phase(
    checkpoints: List[Tuple[str, str]],
    results_dir: str,
    dataset_name: str,
    metadata_results_dir: str,
    progress_callback: Optional[Callable[[List[str]], None]] = None,
) -> None:
    """Chạy một Whisper model chung, batch liên-checkpoint và ưu tiên checkpoint ít ASR nhất."""
    logger.info("Đang pre-load Whisper batched ASR...")
    get_whisper_worker(
        model_name="large-v3",
        language="vi",
        device="cuda",
        compute_type="float16",
        beam_size=5,
        num_workers=WHISPER_NUM_WORKERS,
        word_timestamps=False,
        use_batched_pipeline=True,
    ).load()
    logger.info(
        "Tải xong Whisper. Bắt đầu ASR liên-checkpoint batch_size=%d.",
        WHISPER_BATCH_SIZE,
    )

    states = {}
    for ckpt_name, _ in checkpoints:
        ok_samples = _load_ok_samples(metadata_results_dir, dataset_name, ckpt_name)
        if not ok_samples:
            logger.warning("[%s] Không có sample synthesis ok. Bỏ qua.", ckpt_name)
            continue

        existing, metric_samples = _load_metric_state(
            results_dir, dataset_name, ckpt_name, ASR_METRICS
        )
        to_process = _missing_metric_samples(ok_samples, existing, ASR_METRICS)
        completed = min(len(existing[m]) for m in ASR_METRICS)

        states[ckpt_name] = {
            "ok_total": len(ok_samples),
            "existing": existing,
            "metric_samples": metric_samples,
            "pending": to_process,
            "cursor": 0,
            "initial_completed": completed,
            "completed": completed,
            "next_chart_at": ((completed // CHART_EVERY_SAMPLES) + 1) * CHART_EVERY_SAMPLES,
            "new_processed": 0,
        }

        if to_process:
            logger.info(
                "[%s] ASR resume: đã có %d/%d, còn %d sample.",
                ckpt_name,
                completed,
                len(ok_samples),
                len(to_process),
            )
        else:
            logger.info("[%s] Tất cả %d samples đã có WER/CER (resume).", ckpt_name, len(ok_samples))

    active = {ckpt for ckpt, state in states.items() if state["pending"]}
    if not active:
        logger.info("Tất cả checkpoint đã có WER/CER. Bỏ qua ASR.")
        return

    while active:
        batch_items = []

        while len(batch_items) < WHISPER_BATCH_SIZE and active:
            ckpt_name = min(
                active,
                key=lambda name: (states[name]["completed"], name),
            )
            state = states[ckpt_name]
            idx = state["cursor"]
            pending = state["pending"]

            if idx >= len(pending):
                active.remove(ckpt_name)
                continue

            entry = pending[idx]
            state["cursor"] += 1
            # Cộng trước để batch được phân bố đều giữa các checkpoint ít ASR nhất.
            state["completed"] += 1
            batch_items.append((ckpt_name, entry))

            if state["cursor"] >= len(pending):
                active.remove(ckpt_name)

        if not batch_items:
            break

        batch_entries = [entry for _, entry in batch_items]
        batch_desc = ", ".join(
            f"{name}:{sum(1 for ckpt, _ in batch_items if ckpt == name)}"
            for name in sorted({ckpt for ckpt, _ in batch_items})
        )
        logger.debug("ASR batch %d file: %s", len(batch_entries), batch_desc)
        asr_by_wav = _compute_asr_batch(batch_entries)

        touched = set()
        last_wer_by_ckpt = {}
        for ckpt_name, entry in batch_items:
            state = states[ckpt_name]
            existing = state["existing"]
            metric_samples = state["metric_samples"]
            wav_file = entry["wav_file"]
            asr_vals = asr_by_wav.get(wav_file, {"wer": None, "cer": None, "asr_transcript": ""})

            for metric in ASR_METRICS:
                if wav_file not in existing[metric]:
                    rec = {
                        "wav_file":       wav_file,
                        "value":          asr_vals.get(metric),
                        "asr_transcript": asr_vals.get("asr_transcript", ""),
                    }
                    metric_samples[metric].append(rec)
                    existing[metric][wav_file] = rec

            state["new_processed"] += 1
            touched.add(ckpt_name)
            last_wer_by_ckpt[ckpt_name] = asr_vals.get("wer")

        for ckpt_name in sorted(touched):
            state = states[ckpt_name]
            _save_metric_group(
                results_dir,
                dataset_name,
                ckpt_name,
                ASR_METRICS,
                state["metric_samples"],
            )
            if state["completed"] >= state["next_chart_at"] or state["completed"] == state["ok_total"]:
                if progress_callback is not None:
                    progress_callback(ASR_METRICS)
                state["next_chart_at"] = ((state["completed"] // CHART_EVERY_SAMPLES) + 1) * CHART_EVERY_SAMPLES

            if state["completed"] % LOG_EVERY_SAMPLES == 0 or state["completed"] == state["ok_total"]:
                logger.info(
                    "[%s] ASR progress: %d/%d  WER=%s",
                    ckpt_name,
                    state["completed"],
                    state["ok_total"],
                    f"{last_wer_by_ckpt[ckpt_name]:.3f}" if last_wer_by_ckpt[ckpt_name] is not None else "N/A",
                )

    for ckpt_name, state in sorted(states.items()):
        logger.info(
            "[%s] ASR metrics xong (%d mới, %d resume).",
            ckpt_name,
            state["new_processed"],
            state["initial_completed"],
        )


def run_metric_phase(
    checkpoints: List[Tuple[str, str]],
    results_dir: str,
    dataset_name: str,
    metadata_results_dir: str,
) -> None:
    """Backward-compatible wrapper: chạy audio metrics trước, rồi ASR metrics."""
    run_audio_metric_phase(checkpoints, results_dir, dataset_name, metadata_results_dir)
    run_asr_metric_phase(checkpoints, results_dir, dataset_name, metadata_results_dir)
