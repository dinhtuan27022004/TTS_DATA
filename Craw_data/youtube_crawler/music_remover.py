"""
Phase 3: Music Removal
Loại bỏ nhạc nền từ audio sử dụng Facebook Demucs (htdemucs).
Chỉ giữ lại vocals để chuẩn bị cho transcription.
Sử dụng Multi-processing để chạy song song nhiều file, vắt kiệt VRAM.
Tối ưu: Tự động scale workers để dùng tối đa 14GB VRAM.
"""

import os
import json
import wave
import logging
import multiprocessing as mp
import concurrent.futures
from typing import List

import torch
import torchaudio
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
MAX_VRAM_GB = 14.0  # Giới hạn VRAM tối đa sử dụng (GB)
VRAM_PER_WORKER_GB = 2.5  # Ước lượng VRAM mỗi worker cần (model ~1.1GB + inference buffers ~1.4GB)
SYSTEM_RESERVED_GB = 0.5  # Dành cho Xorg, gnome-shell, etc.

# --- WORKER GLOBALS ---
worker_model = None
worker_device = None

def init_worker(model_name):
    """
    Khởi tạo model Demucs độc lập cho mỗi tiến trình (Process).
    Mỗi tiến trình sẽ có một không gian CUDA (VRAM) riêng.
    """
    global worker_model, worker_device
    import torch
    from demucs.pretrained import get_model

    # Cấu hình CUDA allocator tối ưu
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if torch.cuda.is_available():
        worker_device = torch.device("cuda")
    else:
        worker_device = torch.device("cpu")
        
    worker_model = get_model(model_name)
    worker_model.to(worker_device)
    worker_model.eval()

def process_file_in_worker(args):
    """
    Hàm xử lý tách vocals chạy trong tiến trình con.
    """
    input_path, output_dir = args
    global worker_model, worker_device
    from demucs.apply import apply_model
    import wave
    import gc
    import os

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    try:
        # Lấy thông tin duration
        with wave.open(input_path, 'rb') as wf:
            sr_original = wf.getframerate()
            total_frames = wf.getnframes()
            
        model_sr = worker_model.samplerate

        chunk_duration_s = 10 * 60
        chunk_frames = chunk_duration_s * sr_original
        
        vocals_out = []
        
        for offset in range(0, total_frames, chunk_frames):
            num_frames = min(chunk_frames, total_frames - offset)
            wav, sr = torchaudio.load(input_path, frame_offset=offset, num_frames=num_frames)

            if sr != model_sr:
                wav = torchaudio.functional.resample(wav, sr, model_sr)

            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2, :]

            wav = wav.unsqueeze(0)

            with torch.no_grad():
                sources = apply_model(
                    worker_model, 
                    wav, 
                    device=worker_device,
                    split=True,
                    overlap=0.1,
                    shifts=0,  # Không dùng shifts để giảm memory & tăng tốc
                )

            vocals_idx = worker_model.sources.index("vocals")
            vocals = sources[0, vocals_idx].cpu()
            vocals_out.append(vocals)

            del wav, sources
            if worker_device.type == "cuda":
                torch.cuda.empty_cache()

        final_vocals = torch.cat(vocals_out, dim=1)
        torchaudio.save(output_path, final_vocals, model_sr)

        del final_vocals, vocals_out
        gc.collect()
        if worker_device.type == "cuda":
            torch.cuda.empty_cache()

        return filename, output_path, None
    except torch.cuda.OutOfMemoryError:
        # Cố gắng giải phóng memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return filename, None, "GPU OOM (hết bộ nhớ VRAM cho file này)"
    except Exception as e:
        return filename, None, str(e)


def calculate_optimal_workers() -> int:
    """
    Tính số workers tối ưu dựa trên VRAM khả dụng.
    Target: dùng tối đa MAX_VRAM_GB VRAM.
    """
    if not torch.cuda.is_available():
        return 1
    
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    # Giới hạn VRAM sử dụng
    usable_vram = min(total_vram_gb - SYSTEM_RESERVED_GB, MAX_VRAM_GB)
    
    # Tính số workers tối ưu
    # FP32 model: ~1.1-1.2GB model weight, inference peak ~1.5GB per worker
    optimal_workers = int(usable_vram / VRAM_PER_WORKER_GB)
    
    # Giới hạn bởi số CPU cores
    cpu_count = mp.cpu_count() or 4
    max_by_cpu = max(2, cpu_count - 1)  # Để lại 1 core cho hệ thống
    
    final_workers = max(1, min(optimal_workers, max_by_cpu))
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total VRAM: {total_vram_gb:.1f}GB | Target max: {MAX_VRAM_GB}GB | Usable: {usable_vram:.1f}GB")
    logger.info(f"VRAM per worker (estimated): {VRAM_PER_WORKER_GB}GB")
    logger.info(f"Optimal workers: {optimal_workers} (limited by CPU to {max_by_cpu})")
    
    return final_workers


class MusicRemover:
    def __init__(
        self,
        input_dir: str = "Youtube_Data/Step_0",
        output_dir: str = "Youtube_Data/Step_1",
        model_name: str = "htdemucs",
        chunk_duration: float = 30.0,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.stats_path = os.path.join(output_dir, "stats.json")

    def process_all(self) -> List[str]:
        os.makedirs(self.output_dir, exist_ok=True)

        all_wav_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".wav")]
        all_wav_files.sort()

        if not all_wav_files:
            logger.warning(f"Khong tim thay file WAV nao trong {self.input_dir}")
            return []

        logger.info(f"Tim thay {len(all_wav_files)} file WAV trong {self.input_dir}")

        files_to_process = []
        for filename in all_wav_files:
            output_path = os.path.join(self.output_dir, filename)
            if os.path.exists(output_path):
                logger.info(f"[SKIP] Đã tồn tại: {filename}")
            else:
                files_to_process.append(filename)

        skipped = len(all_wav_files) - len(files_to_process)
        logger.info(f"Da xu ly truoc do: {skipped} files")
        logger.info(f"Can xu ly them: {len(files_to_process)} files")

        if not files_to_process:
            logger.info("Tat ca file da duoc xu ly. Khong can xu ly them.")
            self._save_stats()
            return []

        # Tự động tính số workers tối ưu
        max_workers = calculate_optimal_workers()
        logger.info(f"Khởi chạy {max_workers} tiến trình độc lập (ProcessPool) để tận dụng tối đa ~{MAX_VRAM_GB}GB VRAM...")

        ctx = mp.get_context('spawn')
        args_list = [(os.path.join(self.input_dir, f), self.output_dir) for f in files_to_process]
        processed_files = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=init_worker, initargs=(self.model_name,)) as executor:
            futures = {executor.submit(process_file_in_worker, args): args[0] for args in args_list}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_process), desc="Removing music"):
                fname, output_path, error = future.result()
                if error:
                    logger.error(f"[FAIL] Không thể xử lý {fname}: {error}")
                elif output_path:
                    processed_files.append(output_path)
                    logger.info(f"[OK] {fname}")

        self._save_stats()
        logger.info(f"Hoàn thành! Đã xử lý {len(processed_files)} files mới.")
        return processed_files

    def _save_stats(self):
        total_files = 0
        total_duration = 0.0

        for filename in os.listdir(self.output_dir):
            if not filename.lower().endswith(".wav"):
                continue
            filepath = os.path.join(self.output_dir, filename)
            try:
                with wave.open(filepath, "r") as wf:
                    total_duration += wf.getnframes() / float(wf.getframerate())
                    total_files += 1
            except Exception as e:
                logger.warning(f"Khong the doc duration cua {filename}: {e}")
                total_files += 1

        avg_duration = total_duration / total_files if total_files > 0 else 0.0
        stats = {
            "total_files": total_files,
            "total_duration_seconds": round(total_duration, 2),
            "avg_duration_seconds": round(avg_duration, 2),
        }

        try:
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Stats: {total_files} files, "
                f"{total_duration:.1f}s total, "
                f"{avg_duration:.1f}s avg"
            )
        except Exception as e:
            logger.error(f"Loi luu stats.json: {e}")
