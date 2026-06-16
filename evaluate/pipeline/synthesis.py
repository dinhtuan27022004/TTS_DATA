"""
Phase 1 – TTS Synthesis (multiprocessing).

Mỗi checkpoint chạy trong 1 process riêng (spawn):
  - Model được load DUY NHẤT 1 LẦN tại đầu process
  - Synthesis toàn bộ dataset
  - Lưu WAV + ghi metadata JSON sau mỗi SAVE_EVERY sample (resume-safe)
"""
import logging
import os
import time
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Lưu metadata sau mỗi N sample (để resume được nếu bị ngắt)
SAVE_EVERY = 5


# ─── Worker (chạy trong process riêng – spawn) ────────────────────────────────

def _worker(args: dict) -> None:
    """Synthesis worker.

    Hàm này chạy trong process riêng (spawn). Không được dùng closure/lambda.
    Mọi import cần thiết được thực hiện bên trong hàm.
    """
    import sys
    import logging as _logging
    import os as _os
    import time as _time
    import soundfile as sf

    # Inject project root vào sys.path cho process mới (spawn không kế thừa)
    project_root = args["project_root"]
    f5_src = _os.path.join(project_root, "F5-TTS-Vietnamese", "src")
    custom_tts_src = _os.path.join(project_root, "Custom_TTS", "src")
    for p in (project_root, custom_tts_src, f5_src):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Logging cho process này
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    log = _logging.getLogger(args["ckpt_name"])

    ckpt_name = args["ckpt_name"]
    ckpt_path = args["ckpt_path"]
    samples   = args["samples"]
    out_dir   = args["artifact_dir"]
    meta_path = args["metadata_path"]

    # ── Resume: bỏ qua sample đã tổng hợp xong ───────────────────────────────
    from evaluate.pipeline.persistence import load_synthesis_metadata, save_synthesis_metadata

    existing = load_synthesis_metadata(meta_path)
    done_ids = {e["sample_id"] for e in existing if e.get("status") == "ok"}
    results  = list(existing)
    pending  = [s for s in samples if s["sample_id"] not in done_ids]

    skipped = len(samples) - len(pending)
    if skipped:
        log.info("[%s] Resume: bỏ qua %d sample đã xong.", ckpt_name, skipped)
    if not pending:
        log.info("[%s] Tất cả %d sample đã synthesis. Xong.", ckpt_name, len(samples))
        return

    # ── Load model (1 lần duy nhất) ──────────────────────────────────────────
    log.info("[%s] Đang load model từ %s ...", ckpt_name, ckpt_path)
    try:
        from components.tts.F5_TTS import F5TTSVietnamese
        model = F5TTSVietnamese(ckpt_file=ckpt_path, vocoder_name="vocos", speed=1.0)
    except Exception as exc:
        log.error("[%s] Load model THẤT BẠI: %s", ckpt_name, exc)
        return

    log.info(
        "[%s] Model loaded. Bắt đầu synthesis %d/%d sample...",
        ckpt_name, len(pending), len(samples),
    )

    from tqdm import tqdm
    
    # ── Synthesis loop ────────────────────────────────────────────────────────
    _os.makedirs(out_dir, exist_ok=True)
    ok_count = err_count = 0
    worker_id = args.get("worker_id", 0)

    pbar = tqdm(pending, desc=f"[{ckpt_name}]", position=worker_id, leave=True)
    
    for idx, sample in enumerate(pbar):
        sample_id = sample["sample_id"]
        out_wav   = _os.path.join(out_dir, f"{sample_id}_{ckpt_name}.wav")

        entry: dict = {
            "sample_id":    sample_id,
            "checkpoint":   ckpt_name,
            "wav_file":     _os.path.basename(out_wav),
            "output_path":  out_wav,
            "ref_audio":    sample["wav_path"],
            "ref_text":     sample["text"],
            "gen_text":     sample["text"],
        }

        t0 = _time.time()
        try:
            audio, sr = model.synthesize(
                gen_text      = sample["text"],
                ref_audio_path = sample["wav_path"],
                ref_text      = sample["text"],
            )
            sf.write(out_wav, audio, sr)
            entry["synth_time_s"] = round(_time.time() - t0, 3)
            entry["sample_rate"]  = int(sr)
            entry["status"]       = "ok"
            ok_count += 1
        except Exception as exc:
            entry["status"] = "error"
            entry["error"]  = str(exc)
            err_count += 1
            # In lỗi không làm hỏng pbar bằng tqdm.write
            tqdm.write(f"[{ckpt_name}] Lỗi {sample_id}: {exc}")

        results.append(entry)
        pbar.set_postfix(ok=ok_count, err=err_count)

        # Lưu metadata định kỳ để đảm bảo resume được
        if (idx + 1) % SAVE_EVERY == 0 or (idx + 1) == len(pending):
            save_synthesis_metadata(meta_path, results)

    log.info("[%s] Synthesis xong: %d ok, %d lỗi.", ckpt_name, ok_count, err_count)


# ─── Phase 1 entry point ──────────────────────────────────────────────────────

def run_synthesis_phase(
    checkpoints: List[Tuple[str, str]],
    samples: List[dict],
    artifact_dir: str,
    metadata_results_dir: str,
    dataset_name: str,
    num_workers: int = 2,
    project_root: str = "",
) -> None:
    """Chạy Phase 1: Synthesis song song bằng multiprocessing (spawn).

    Args:
        checkpoints:          List (ckpt_name, ckpt_path).
        samples:              List sample dicts từ discovery.
        artifact_dir:         Thư mục lưu WAV tổng hợp.
        metadata_results_dir: Thư mục lưu metadata JSON.
        dataset_name:         Tên dataset (basename).
        num_workers:          Số process song song tối đa.
        project_root:         Root dir dự án (inject vào sys.path của worker).
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    # Build args cho từng worker
    worker_args_list = []
    for ckpt_name, ckpt_path in checkpoints:
        meta_path = os.path.join(
            metadata_results_dir,
            f"{dataset_name}_{ckpt_name}_metadata.json",
        )
        worker_args_list.append(
            {
                "ckpt_name":    ckpt_name,
                "ckpt_path":    ckpt_path,
                "samples":      samples,
                "artifact_dir": artifact_dir,
                "metadata_path": meta_path,
                "project_root": project_root,
                "worker_id":    len(worker_args_list),
            }
        )

    n_parallel = min(num_workers, len(checkpoints))
    logger.info(
        "Phase 1 – Synthesis: %d checkpoint, %d worker song song, %d samples/checkpoint.",
        len(checkpoints), n_parallel, len(samples),
    )

    # Chạy batch: tối đa num_workers process cùng lúc
    active: list = []
    queue = list(worker_args_list)

    while queue or active:
        # Khởi động worker mới nếu còn slot
        while queue and len(active) < num_workers:
            args = queue.pop(0)
            p = ctx.Process(target=_worker, args=(args,), daemon=False)
            p.start()
            active.append((p, args["ckpt_name"]))
            logger.info("  → Worker khởi động: %s (pid=%d)", args["ckpt_name"], p.pid)

        time.sleep(3)

        # Kiểm tra worker nào đã xong
        still_alive = []
        for p, name in active:
            if p.is_alive():
                still_alive.append((p, name))
            else:
                p.join()
                status = "OK" if p.exitcode == 0 else f"EXIT({p.exitcode})"
                logger.info("  ← Worker xong: %s [%s]", name, status)
        active = still_alive

    logger.info("Phase 1 hoàn thành.")
