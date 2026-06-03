#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepFilterNet Speech Enhancement — Worker Respawn Architecture

Tại sao RAM luôn tăng dù đã gc/del/malloc_trim?
  - Python pymalloc arena: không bao giờ trả OS
  - glibc heap fragmentation: lỗ giữa heap không trim được
  - PyTorch CUDA pool: cache ngay cả sau empty_cache()
  → Không có cách nào giữ RAM ổn định trong 1 process chạy lâu dài.

Giải pháp: Worker xử lý BATCH_SIZE file rồi EXIT.
  → OS thu hồi 100% RAM ngay lập tức, không có fragmentation.
  → Main process spawn worker mới cho batch tiếp theo.
  → RAM luôn ổn định ở mức 1 batch.
"""

# ==============================================================================
# HACK: torchaudio 2.11+ compatibility monkeypatch
# Torchaudio 2.11+ routes torchaudio.load through torchcodec, which is stricter
# and crashes on malformed/corrupted audio files. We replace both .info and .load
# with soundfile-based implementations that are more permissive (matches old
# sox/soundfile backend behavior).
# ==============================================================================
import sys
import torchaudio
from types import ModuleType
import soundfile as sf
import numpy as np
import torch

class AudioMetaData:
    def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample=16, encoding="PCM_S"):
        self.sample_rate = sample_rate; self.num_frames = num_frames
        self.num_channels = num_channels; self.bits_per_sample = bits_per_sample
        self.encoding = encoding

def mock_info(filepath, *args, **kwargs):
    try:
        info = sf.info(filepath)
        return AudioMetaData(info.samplerate, info.frames, info.channels)
    except Exception as e:
        raise RuntimeError(f"Cannot read audio metadata: {e}") from e

def mock_load(filepath, frame_offset=0, num_frames=-1, normalize=True,
              channels_first=True, format=None, buffer_size=65536, **kwargs):
    """soundfile-backed torchaudio.load — avoids torchcodec for corrupt files."""
    sf_kwargs = {}
    if frame_offset > 0:
        sf_kwargs["start"] = frame_offset
    if num_frames > 0:
        sf_kwargs["stop"] = frame_offset + num_frames
    data, sr = sf.read(filepath, dtype="float32", always_2d=True, **sf_kwargs)
    # data shape: (frames, channels) → convert to (channels, frames)
    tensor = torch.from_numpy(data.T)  # (C, T)
    if not normalize:
        # soundfile always returns float32 in [-1, 1]; undo norm for int types
        info = sf.info(filepath)
        if "PCM" in info.subtype:
            bits = int(''.join(filter(str.isdigit, info.subtype)) or 16)
            tensor = (tensor * (2 ** (bits - 1))).to(torch.int16 if bits <= 16 else torch.int32)
    return tensor, sr

torchaudio.info = mock_info
torchaudio.load = mock_load
_m = ModuleType("torchaudio.backend.common")
_m.AudioMetaData = AudioMetaData
sys.modules["torchaudio.backend.common"] = _m
# ==============================================================================

import os
import sys
import logging
import warnings
import gc
import multiprocessing
import queue
import json
import tempfile
import datetime
from tqdm import tqdm

os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "65536")
os.environ.setdefault("MALLOC_MMAP_MAX_",       "65536")

import torch

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "Processed_DATA", "PhoAudioBook")

# Số worker chạy song song
NUM_PROCESSES = 8  # Tối ưu cho 1 GPU: >4 worker tranh GPU → chậm hơn (11→7 it/s với 8 workers)

# Số file mỗi worker xử lý trước khi EXIT để OS thu hồi RAM.
# JANGAN terlalu kecil: mỗi lần respawn mất ~30s (init_df load checkpoint).
# Với 888K files, 4 workers, 11 it/s:
#   BATCH_SIZE=5000 → ~30 phút/batch, ~44 lần respawn tổng → overhead ~3%
#   BATCH_SIZE=200  → ~2 phút/batch, ~1111 lần respawn → overhead >50% (CHẬM!)
BATCH_SIZE = 50

TMP_DIR           = "/home/reg/TTS_DATA/TMP"
STATE_FILE        = os.path.join(BASE_DIR, "Processed_DATA", "preprocess_state.json")
CORRUPTED_LOG     = os.path.join(BASE_DIR, "Processed_DATA", "corrupted_files.txt")

# Các keyword trong exception message được coi là "file corrupt" → chỉ log WARNING ngắn
_CORRUPT_KEYWORDS = ("format not recognised", "invalid data", "no such file",
                     "cannot read audio", "failed to decode")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# STATE FILE
# ==============================================================================
def load_preprocess_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Lỗi đọc state: {e}")
        return {}

def update_preprocess_state(path, key, last_file, success_count, error_count):
    state = load_preprocess_state(path)
    state[key] = {
        "last_processed_file": last_file,
        "success_count": success_count,
        "error_count":   error_count,
        "updated_at":    datetime.datetime.now().isoformat()
    }
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False,
                                         suffix=".tmp", encoding="utf-8") as tf:
            json.dump(state, tf, ensure_ascii=False, indent=4)
            tmp = tf.name
        os.replace(tmp, path)
    except Exception as e:
        logger.error(f"Lỗi ghi state: {e}")

# ==============================================================================
# WORKER — xử lý file_list rồi EXIT
# OS sẽ thu hồi 100% RAM của process này khi nó kết thúc.
# ==============================================================================
_DONE = None  # sentinel

def worker_task(shared_model, file_list, process_idx, result_queue):
    """
    Xử lý toàn bộ file_list rồi exit.
    Không cần gc/malloc_trim — process exit là cách duy nhất
    đảm bảo RAM trả về OS hoàn toàn.
    """
    from df.enhance import init_df, enhance, load_audio

    try:
        _tmp_model, df_state, _ = init_df()
        del _tmp_model  # dùng shared_model
        gc.collect()
    except Exception as e:
        logger.error(f"[W{process_idx}] Không thể init df_state: {e}")
        for fp in file_list:
            result_queue.put((fp, False))
        result_queue.put(_DONE)
        return

    df_sr = df_state.sr()

    for idx, filepath in enumerate(file_list):
        audio = enhanced = enhanced_resampled = None
        try:
            orig_sr = torchaudio.info(filepath).sample_rate
            audio, _ = load_audio(filepath, sr=df_sr)

            with torch.no_grad():
                enhanced = enhance(shared_model, df_state, audio)
            enhanced = enhanced.cpu()

            if orig_sr != df_sr:
                enhanced_resampled = torchaudio.functional.resample(enhanced, df_sr, orig_sr)
            else:
                enhanced_resampled = enhanced.clone()
            enhanced_resampled = enhanced_resampled.detach()

            # Lưu sample nghe thử
            if idx == 0:
                os.makedirs(TMP_DIR, exist_ok=True)
                base = os.path.splitext(os.path.basename(filepath))[0]
                orig_wav, _ = torchaudio.load(filepath)
                torchaudio.save(os.path.join(TMP_DIR, f"{base}_original.wav"),  orig_wav,           orig_sr)
                torchaudio.save(os.path.join(TMP_DIR, f"{base}_processed.wav"), enhanced_resampled, orig_sr)
                del orig_wav

            torchaudio.save(filepath, enhanced_resampled, orig_sr)
            result_queue.put((filepath, True))

        except Exception as exc:
            exc_lower = str(exc).lower()
            if any(kw in exc_lower for kw in _CORRUPT_KEYWORDS):
                # File bị corrupt / không đọc được → skip, log ngắn
                logger.warning(f"[W{process_idx}] SKIP corrupt file: {os.path.basename(filepath)} ({exc})")
                try:
                    with open(CORRUPTED_LOG, "a", encoding="utf-8") as cf:
                        cf.write(filepath + "\n")
                except Exception:
                    pass
            else:
                # Lỗi không mong đợi → log full traceback
                import traceback
                logger.error(f"[W{process_idx}] {filepath}: {exc}\n{traceback.format_exc()}")
            result_queue.put((filepath, False))

        finally:
            for t in [audio, enhanced, enhanced_resampled]:
                if t is not None:
                    del t
            # Không cần gc/malloc_trim — process sắp exit sẽ dọn sạch hơn

    result_queue.put(_DONE)
    # Process exit → OS thu hồi toàn bộ RAM ngay lập tức

# ==============================================================================
# BATCH RUNNER — spawn workers theo batch, đợi xong, spawn tiếp
# ==============================================================================
def run_batch(shared_model, batch_files, batch_start_idx, pbar,
              success_count, error_count, last_processed):
    """
    Chia batch thành NUM_PROCESSES chunk, spawn workers, đợi tất cả xong.
    Khi tất cả workers exit → RAM sạch hoàn toàn.
    """
    chunks = [batch_files[i::NUM_PROCESSES] for i in range(NUM_PROCESSES)]
    result_queue = multiprocessing.Queue(maxsize=NUM_PROCESSES * 50)
    processes = []

    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = multiprocessing.Process(
            target=worker_task,
            args=(shared_model, chunk, i, result_queue)
        )
        processes.append(p)
        p.start()

    num_workers = len(processes)
    completed_files = set()
    prefix_idx = 0
    workers_done = 0

    try:
        while workers_done < num_workers:
            try:
                item = result_queue.get(timeout=1.0)
                if item is _DONE:
                    workers_done += 1
                    continue

                filepath, success = item
                if success: success_count += 1
                else:       error_count   += 1

                completed_files.add(filepath)
                # Tìm file liên tục cuối cùng để update state
                while (prefix_idx < len(batch_files) and
                       batch_files[prefix_idx] in completed_files):
                    prefix_idx += 1

                if prefix_idx > 0:
                    last_consecutive = batch_files[prefix_idx - 1]
                    rel = os.path.relpath(last_consecutive, INPUT_DIR)
                    update_preprocess_state(STATE_FILE, "denoise_deepfilter",
                                            rel, success_count, error_count)
                    last_processed = rel

                pbar.update(1)

            except queue.Empty:
                # Kiểm tra worker còn sống không
                if all(not p.is_alive() for p in processes):
                    logger.warning("Tất cả workers đã thoát sớm.")
                    break

    except KeyboardInterrupt:
        logger.warning("Ctrl+C — terminate workers...")
        for p in processes: p.terminate()
        for p in processes: p.join(timeout=5)
        raise

    # Join tất cả workers — đảm bảo OS đã thu hồi RAM trước khi spawn batch mới
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            logger.warning(f"Worker {p.pid} không phản hồi, terminate...")
            p.terminate()
            p.join()

    return success_count, error_count, last_processed

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not os.path.isdir(INPUT_DIR):
        logger.error(f"Thư mục không tồn tại: {INPUT_DIR}")
        sys.exit(1)

    # Load model 1 lần, share memory cho tất cả workers
    logger.info("Đang tải model DeepFilterNet...")
    from df.enhance import init_df
    shared_model, _, _ = init_df()
    shared_model.eval()
    shared_model.share_memory()
    logger.info("Model sẵn sàng (pinned to shared memory).")
    gc.collect()

    # Resume state
    state          = load_preprocess_state(STATE_FILE)
    df_resume      = state.get("denoise_deepfilter", {})
    last_processed = df_resume.get("last_processed_file")
    success_count  = df_resume.get("success_count", 0)
    error_count    = df_resume.get("error_count",   0)

    if last_processed:
        logger.info(f"Resume từ: {last_processed} (đã xử lý: {success_count + error_count})")
    else:
        logger.info("Bắt đầu từ đầu.")

    # Quét tất cả file WAV
    all_wav_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(INPUT_DIR)
        for f in files if f.lower().endswith(".wav")
    )

    pending = [
        fp for fp in all_wav_files
        if not (last_processed and os.path.relpath(fp, INPUT_DIR) <= last_processed)
    ]

    if not pending:
        logger.info("Không còn file cần xử lý. Hoàn tất.")
        return

    total = len(pending)
    logger.info(f"Tổng số file cần xử lý: {total} | Batch size: {BATCH_SIZE} file/worker-cycle")
    logger.info(f"Số worker: {NUM_PROCESSES} | Ước tính số batch: {-(-total // (BATCH_SIZE * NUM_PROCESSES))}")

    try:
        with tqdm(total=total, desc="DeepFilterNet Denoise") as pbar:
            # Chia thành các batch lớn (BATCH_SIZE × NUM_PROCESSES file/batch)
            batch_total = BATCH_SIZE * NUM_PROCESSES
            for batch_start in range(0, total, batch_total):
                batch = pending[batch_start : batch_start + batch_total]
                success_count, error_count, last_processed = run_batch(
                    shared_model, batch, batch_start, pbar,
                    success_count, error_count, last_processed
                )
                # Sau khi run_batch() return: tất cả worker processes đã exit
                # → OS đã thu hồi RAM → bắt đầu batch mới với RAM sạch

    except KeyboardInterrupt:
        logger.warning("Đã dừng. Chạy lại để resume.")
        sys.exit(1)

    logger.info("=== HOÀN TẤT ===")
    logger.info(f"Thành công: {success_count} | Thất bại: {error_count}")
    logger.info(f"Samples tại: {TMP_DIR}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
