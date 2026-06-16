"""
F5-TTS Evaluation Pipeline – Entry Point.

Chỉnh các tham số trong phần CONFIG bên dưới, sau đó chạy:
    python run_eval_pipeline.py

3 giai đoạn tự động:
  Phase 1 – Synthesis: N process song song (multiprocessing spawn)
  Phase 2A – ASR Metrics:   Whisper Large v3 chạy song song ở thread nền
  Phase 2B – Audio Metrics: STOI/PESQ tuần tự, F0/UTMOS/SpeakerSim song song có kiểm soát
  Phase 2C – Audio Charts:  metric nào xong thì vẽ chart metric đó
  Phase 3  – Final Charts:  chờ Whisper xong rồi vẽ chart tổng hợp

Mọi giai đoạn đều có cơ chế resume:
  - Phase 1: bỏ qua WAV đã tổng hợp (dựa vào metadata JSON)
  - Phase 2: bỏ qua sample đã tính metric (dựa vào JSON trong evaluate/results/)
"""
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════
#  CẤU HÌNH – Chỉnh trực tiếp tại đây
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Dataset ──────────────────────────────────────────────────
# Thư mục chứa các cặp file .wav + .txt (có thể nested)
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "YTDT2")

# ── Checkpoints ──────────────────────────────────────────────
# None = tự động detect theo thứ tự: v0 → 50000 → 60000 → 70000
# Hoặc chỉ định tên cụ thể, ví dụ:
#   CHECKPOINTS = ["f5-tts-70000", "f5-tts-v0"]
CHECKPOINTS = ["f5-tts-v0", "f5-tts-50000", "f5-tts-60000", "f5-tts-70000", "f5-tts-last"]

# ── Synthesis ─────────────────────────────────────────────────
# Số checkpoint chạy song song (mỗi cái load 1 model riêng vào GPU)
NUM_WORKERS = 2

# ── Test mode ─────────────────────────────────────────────────
# True  → chỉ chạy TEST_NUM_SAMPLES sample đầu (để kiểm tra nhanh)
# False → chạy toàn bộ dataset
TEST_MODE        = True
TEST_NUM_SAMPLES = 1000

# ── Output dirs ───────────────────────────────────────────────
ARTIFACT_DIR  = os.path.join(PROJECT_ROOT, "artifact")              # WAV tổng hợp
ARTIFACT_META = os.path.join(PROJECT_ROOT, "artifact", "results")   # Metadata synthesis
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "evaluate", "results")   # Metric JSON
CHARTS_DIR    = os.path.join(PROJECT_ROOT, "evaluate", "charts")    # Seaborn charts

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ═══════════════════════════════════════════════════════════════

# Setup sys.path (cũng sẽ được inject vào PYTHONPATH cho worker process)
_F5_SRC = os.path.join(PROJECT_ROOT, "F5-TTS-Vietnamese", "src")
for _p in (PROJECT_ROOT, _F5_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Inject PYTHONPATH để spawned process kế thừa
_existing_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    f"{PROJECT_ROOT}:{_F5_SRC}" + (f":{_existing_pp}" if _existing_pp else "")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _print_summary(checkpoints, results_dir, dataset_name):
    """In bảng tổng kết metric mean sau khi hoàn thành."""
    import json

    METRICS  = ["pesq", "stoi", "utmos", "f0_corr", "speaker_sim", "wer", "cer"]
    col_w    = 13
    sep      = "─" * (18 + col_w * len(METRICS))

    header = f"{'Checkpoint':<18}" + "".join(f"{m.upper():>{col_w}}" for m in METRICS)
    print(f"\n{sep}")
    print("  BẢNG TỔNG KẾT (giá trị trung bình)")
    print(sep)
    print(f"  {header}")
    print(sep)

    for ckpt_name, _ in checkpoints:
        row = f"  {ckpt_name:<16}"
        for metric in METRICS:
            path = os.path.join(results_dir, dataset_name, ckpt_name, f"{metric}.json")
            val  = None
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    val = data.get("summary", {}).get("mean")
                except Exception:
                    pass
            row += f"{val:>{col_w}.3f}" if val is not None else f"{'N/A':>{col_w}}"
        print(row)

    print(sep + "\n")


def main():
    t_start = time.time()

    from evaluate.pipeline.discovery    import get_checkpoints, get_dataset_samples
    from evaluate.pipeline.synthesis    import run_synthesis_phase
    from evaluate.pipeline.metric_runner import (
        AUDIO_METRIC_ORDER,
        AUDIO_METRICS,
        run_asr_metric_phase,
        run_audio_metric_phase,
    )
    from evaluate.pipeline.visualizer   import plot_comparison

    logger.info("=" * 60)
    logger.info("  F5-TTS Evaluation Pipeline")
    logger.info("=" * 60)

    # ── Discovery ─────────────────────────────────────────────────────────────
    checkpoints = get_checkpoints(MODELS_DIR, CHECKPOINTS)
    if not checkpoints:
        logger.error("Không tìm thấy checkpoint. Kiểm tra thư mục: %s", MODELS_DIR)
        sys.exit(1)

    samples = get_dataset_samples(DATASET_PATH)
    if not samples:
        logger.error("Không tìm thấy sample. Kiểm tra dataset: %s", DATASET_PATH)
        sys.exit(1)

    if TEST_MODE:
        samples = samples[:TEST_NUM_SAMPLES]
        logger.info("⚠  TEST MODE – chỉ chạy %d sample đầu.", TEST_NUM_SAMPLES)

    dataset_name = os.path.basename(os.path.normpath(DATASET_PATH))
    artifact_dir = os.path.join(ARTIFACT_DIR, dataset_name)

    logger.info("Dataset     : %s (%d samples)", dataset_name, len(samples))
    logger.info("Checkpoints : %s", [n for n, _ in checkpoints])
    logger.info("WAV output  : %s", artifact_dir)
    logger.info("Metric JSON : %s", RESULTS_DIR)

    last_chart_counts = {}
    chart_lock = threading.Lock()

    def plot_metric_progress(metric_keys):
        # matplotlib/seaborn are not thread-safe; serialize live chart writes.
        with chart_lock:
            key = tuple(metric_keys)
            last_chart_counts[key] = last_chart_counts.get(key, 0) + 1
            suffix = "_" + "_".join(metric_keys) + "_live"
            title = " – " + "/".join(metric_keys) + " live"
            charts = plot_comparison(
                RESULTS_DIR,
                dataset_name,
                CHARTS_DIR,
                metric_keys=list(metric_keys),
                filename_suffix=suffix,
                title_suffix=title,
            )
            if last_chart_counts[key] % 10 == 1:
                for c in charts:
                    logger.info("  Live chart updated: %s", c)

    # ── Phase 1: Synthesis ────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Phase 1 – TTS Synthesis  (%d workers song song)", NUM_WORKERS)
    run_synthesis_phase(
        checkpoints          = checkpoints,
        samples              = samples,
        artifact_dir         = artifact_dir,
        metadata_results_dir = ARTIFACT_META,
        dataset_name         = dataset_name,
        num_workers          = NUM_WORKERS,
        project_root         = PROJECT_ROOT,
    )

    # ── Phase 2: ASR chạy song song với audio metrics ────────────────────────
    logger.info("")
    logger.info("▶ Phase 2 – chạy Whisper song song với các audio metrics")

    controlled_parallel_metrics = ["f0_corr", "utmos", "speaker_sim"]
    sequential_audio_metrics = [m for m in AUDIO_METRIC_ORDER if m not in controlled_parallel_metrics]

    with ThreadPoolExecutor(max_workers=4) as metric_executor:
        asr_future = metric_executor.submit(
            run_asr_metric_phase,
            checkpoints,
            RESULTS_DIR,
            dataset_name,
            ARTIFACT_META,
            plot_metric_progress,
        )
        logger.info("Đã khởi động thread Whisper/WER/CER ở nền.")

        logger.info("")
        logger.info("▶ Phase 2B – Audio Metrics nhẹ chạy tuần tự")
        completed_audio_metrics = []
        for metric in sequential_audio_metrics:
            logger.info("")
            logger.info("▶ Audio metric – %s", metric.upper())
            run_audio_metric_phase(
                checkpoints          = checkpoints,
                results_dir          = RESULTS_DIR,
                dataset_name         = dataset_name,
                metadata_results_dir = ARTIFACT_META,
                metrics              = [metric],
                progress_callback    = plot_metric_progress,
            )

            completed_audio_metrics.append(metric)
            with chart_lock:
                metric_charts = plot_comparison(
                    RESULTS_DIR,
                    dataset_name,
                    CHARTS_DIR,
                    metric_keys=[metric],
                    filename_suffix=f"_{metric}",
                    title_suffix=f" – {metric}",
                )
            for c in metric_charts:
                logger.info("  %s chart: %s", metric.upper(), c)

        logger.info("")
        logger.info("▶ Phase 2C – Audio Metrics nặng chạy song song có kiểm soát: %s", controlled_parallel_metrics)
        audio_futures = {
            metric_executor.submit(
                run_audio_metric_phase,
                checkpoints,
                RESULTS_DIR,
                dataset_name,
                ARTIFACT_META,
                [metric],
                plot_metric_progress,
            ): metric
            for metric in controlled_parallel_metrics
        }

        for future in as_completed(audio_futures):
            metric = audio_futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error("Audio metric %s lỗi: %s", metric, exc, exc_info=True)
                continue

            completed_audio_metrics.append(metric)
            with chart_lock:
                metric_charts = plot_comparison(
                    RESULTS_DIR,
                    dataset_name,
                    CHARTS_DIR,
                    metric_keys=[metric],
                    filename_suffix=f"_{metric}",
                    title_suffix=f" – {metric}",
                )
            for c in metric_charts:
                logger.info("  %s chart: %s", metric.upper(), c)

        logger.info("")
        logger.info("▶ Phase 2D – Audio Metric Charts tổng hợp")
        with chart_lock:
            audio_charts = plot_comparison(
                RESULTS_DIR,
                dataset_name,
                CHARTS_DIR,
                metric_keys=completed_audio_metrics or AUDIO_METRICS,
                filename_suffix="_audio",
                title_suffix=" – audio metrics",
            )
        for c in audio_charts:
            logger.info("  Audio chart: %s", c)

        logger.info("")
        logger.info("Đợi thread Whisper/WER/CER hoàn tất trước khi vẽ chart tổng hợp cuối...")
        asr_future.result()

    # ── Phase 3: Final visualization ─────────────────────────────────────────
    logger.info("")
    logger.info("▶ Phase 3 – Final Visualization  (audio + WER/CER)")
    charts = plot_comparison(RESULTS_DIR, dataset_name, CHARTS_DIR)
    for c in charts:
        logger.info("  Chart: %s", c)

    # ── Summary table ─────────────────────────────────────────────────────────
    _print_summary(checkpoints, RESULTS_DIR, dataset_name)

    elapsed = time.time() - t_start
    logger.info("Hoàn thành! Tổng thời gian: %.1f giây (%.1f phút)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
