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
        self.processed_json_path = os.path.join(output_dir, "processed_demucs.json")

    def _load_processed_files(self) -> set:
        if os.path.exists(self.processed_json_path):
            try:
                with open(self.processed_json_path, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Không thể đọc file {self.processed_json_path}: {e}")
        return set()

    def _save_processed_files(self, processed_set: set):
        try:
            with open(self.processed_json_path, "w", encoding="utf-8") as f:
                json.dump(list(processed_set), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi lưu file {self.processed_json_path}: {e}")

    def process_all(self) -> List[str]:
        import time
        os.makedirs(self.output_dir, exist_ok=True)
        
        processed_set = self._load_processed_files()
        if processed_set:
            logger.info(f"Đã nạp {len(processed_set)} files đã xử lý từ file JSON.")
        
        total_processed_in_session = []
        max_empty_retries = 3
        empty_retries = 0

        # Chỉ chạy 1 luồng duy nhất theo yêu cầu để tránh lỗi hết RAM (OOM)
        max_workers = 1
        logger.info(f"Khởi chạy {max_workers} tiến trình độc lập (ProcessPool)...")

        while True:
            all_wav_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".wav")]
            all_wav_files.sort()

            files_to_process = []
            json_updated = False
            for filename in all_wav_files:
                output_path = os.path.join(self.output_dir, filename)
                # Đã có trong JSON thì an toàn bỏ qua
                if filename in processed_set:
                    continue
                # Nếu file output đã tồn tại (do lần chạy trước bị crash chưa kịp lưu JSON)
                if os.path.exists(output_path):
                    processed_set.add(filename)
                    json_updated = True
                    continue
                files_to_process.append(filename)
            
            if json_updated:
                self._save_processed_files(processed_set)

            if not files_to_process:
                empty_retries += 1
                if empty_retries > max_empty_retries:
                    logger.info("Không có file mới nào sau nhiều lần thử. Kết thúc Phase 3.")
                    break
                logger.info(f"Chưa có file mới. Chờ 60 giây và thử lại... ({empty_retries}/{max_empty_retries})")
                time.sleep(60)
                continue

            # Reset retries vì đã tìm thấy file mới
            empty_retries = 0
            logger.info(f"Cần xử lý thêm lô mới: {len(files_to_process)} files")

            ctx = mp.get_context('spawn')
            args_list = [(os.path.join(self.input_dir, f), self.output_dir) for f in files_to_process]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=init_worker, initargs=(self.model_name,)) as executor:
                futures = {executor.submit(process_file_in_worker, args): args[0] for args in args_list}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_process), desc="Removing music"):
                    fname, output_path, error = future.result()
                    if error:
                        logger.error(f"[FAIL] Không thể xử lý {fname}: {error}")
                    elif output_path:
                        total_processed_in_session.append(output_path)
                        processed_set.add(fname)
                        logger.info(f"[OK] {fname}")
            
            # Lưu lại danh sách JSON sau mỗi lô
            self._save_processed_files(processed_set)
            self._save_stats()

        logger.info(f"Hoàn thành! Đã xử lý tổng cộng {len(total_processed_in_session)} files mới trong phiên này.")
        return total_processed_in_session

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
