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
import numpy as np
import soundfile as sf
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
    import gc
    import os

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    try:
        model_sr = worker_model.samplerate

        logger.info(f"[{filename}] Đang load toàn bộ audio vào RAM...")
        wav, sr = torchaudio.load(input_path)

        if sr != model_sr:
            wav = torchaudio.functional.resample(wav, sr, model_sr)

        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]

        wav = wav.unsqueeze(0)

        logger.info(f"[{filename}] Bắt đầu tách nhạc (Demucs tự động chia chunk)...")
        with torch.no_grad():
            sources = apply_model(
                worker_model, 
                wav, 
                device=worker_device,
                split=True,
                overlap=0.1,
                shifts=0,
                progress=True,
            )

        vocals_idx = worker_model.sources.index("vocals")
        vocals = sources[0, vocals_idx].cpu()
        
        torchaudio.save(output_path, vocals, model_sr)

        del wav, sources, vocals
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

            # 1. KIỂM TRA FILE ĐẦU TIÊN: Nếu là file gốc lớn hơn 1 tiếng thì chia nhỏ trước khi xử lý
            first_file = files_to_process[0]
            if "_chunk_" not in first_file:
                filepath = os.path.join(self.input_dir, first_file)
                try:
                    info = sf.info(filepath)
                    duration = info.duration
                    if duration > 3600:
                        logger.info(f"Phát hiện file gốc lớn hơn 1 tiếng: {first_file} ({duration:.1f}s). Bắt đầu chia nhỏ tại khoảng lặng...")
                        split_samples, sr = self._find_split_points(filepath)
                        self._split_audio(filepath, split_samples, sr)
                        
                        # Xóa file gốc khỏi Step_0 để hệ thống chuyển sang xử lý các file chunk
                        os.remove(filepath)
                        processed_set.add(first_file) # Đánh dấu đã xử lý file gốc
                        self._save_processed_files(processed_set)
                        logger.info(f"Đã chia nhỏ và xóa file gốc {first_file} khỏi {self.input_dir}. Khởi động lại vòng lặp để xử lý các file chunk...")
                        continue # Restart loop to pick up the new chunk files
                except Exception as e:
                    logger.error(f"Lỗi khi kiểm tra/chia nhỏ file {first_file}: {e}")

            # 2. Xử lý đúng 1 file/chunk đầu tiên trong hàng đợi để kiểm soát bộ nhớ
            file_to_run = files_to_process[0]
            logger.info(f"Bắt đầu xử lý file: {file_to_run}")

            ctx = mp.get_context('spawn')
            args_list = [(os.path.join(self.input_dir, file_to_run), self.output_dir)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=init_worker, initargs=(self.model_name,)) as executor:
                futures = {executor.submit(process_file_in_worker, args): args[0] for args in args_list}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(args_list), desc="Removing music"):
                    fname, output_path, error = future.result()
                    if error:
                        logger.error(f"[FAIL] Không thể xử lý {fname}: {error}")
                    elif output_path:
                        total_processed_in_session.append(output_path)
                        processed_set.add(fname)
                        logger.info(f"[OK] {fname}")
                        
                        # Xóa file gốc/chunk sau khi xử lý thành công
                        input_file_path = os.path.join(self.input_dir, fname)
                        if os.path.exists(input_file_path):
                            try:
                                os.remove(input_file_path)
                                logger.info(f"[DELETED] {fname} từ {self.input_dir}")
                            except Exception as e:
                                logger.error(f"[FAIL] Lỗi khi xóa {fname}: {e}")
            
            # Lưu lại danh sách JSON sau mỗi file
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

    def _find_split_points(self, audio_path, min_silence_len_sec=2.0, silence_thresh_db=-45, max_chunk_len_sec=900):
        """
        Tìm tất cả các khoảng lặng >= min_silence_len_sec và cắt tại trung điểm của chúng.
        Nếu khoảng cách giữa 2 điểm cắt vượt quá max_chunk_len_sec, tiến hành cắt cưỡng bức tại điểm yên tĩnh nhất.
        """
        info = sf.info(audio_path)
        sr = info.samplerate
        total_samples = info.frames
        
        block_size = int(sr * 0.1) # 100ms blocks
        block_duration = block_size / sr
        min_silence_blocks = int(min_silence_len_sec / block_duration)
        max_chunk_blocks = int(max_chunk_len_sec / block_duration)
        
        rms_list = []
        with sf.SoundFile(audio_path) as f:
            for block in f.blocks(blocksize=block_size, overlap=0, always_2d=True):
                mono_block = np.mean(block, axis=1)
                rms = np.sqrt(np.mean(mono_block**2))
                rms_list.append(rms)
                
        rms_arr = np.array(rms_list)
        rms_arr = np.clip(rms_arr, 1e-10, None)
        db_arr = 20 * np.log10(rms_arr)
        
        is_silent = db_arr < silence_thresh_db
        
        # Tìm các khoảng lặng liên tục
        silent_intervals = []
        in_silence = False
        silence_start = 0
        
        for i, silent in enumerate(is_silent):
            if silent:
                if not in_silence:
                    in_silence = True
                    silence_start = i
            else:
                if in_silence:
                    in_silence = False
                    silence_len = i - silence_start
                    if silence_len >= min_silence_blocks:
                        silent_intervals.append((silence_start, i))
                        
        if in_silence:
            silence_len = len(is_silent) - silence_start
            if silence_len >= min_silence_blocks:
                silent_intervals.append((silence_start, len(is_silent)))
                
        # Xác định điểm cắt ở giữa các khoảng lặng
        split_blocks = [0]
        last_split_block = 0
        
        for start_block, end_block in silent_intervals:
            mid_block = (start_block + end_block) // 2
            
            # Cắt cưỡng bức nếu khoảng cách quá dài
            if mid_block - last_split_block > max_chunk_blocks:
                temp_last = last_split_block
                while mid_block - temp_last > max_chunk_blocks:
                    search_start = temp_last + int(max_chunk_blocks * 0.8)
                    search_end = min(temp_last + max_chunk_blocks, mid_block)
                    if search_start >= search_end:
                        break
                    best_sub_block = np.argmin(db_arr[search_start:search_end]) + search_start
                    split_blocks.append(best_sub_block)
                    temp_last = best_sub_block
                    
            split_blocks.append(mid_block)
            last_split_block = split_blocks[-1]
            
        # Kiểm tra đoạn cuối cùng
        total_blocks = len(db_arr)
        if total_blocks - last_split_block > max_chunk_blocks:
            temp_last = last_split_block
            while total_blocks - temp_last > max_chunk_blocks:
                search_start = temp_last + int(max_chunk_blocks * 0.8)
                search_end = min(temp_last + max_chunk_blocks, total_blocks)
                if search_start >= search_end:
                    break
                best_sub_block = np.argmin(db_arr[search_start:search_end]) + search_start
                split_blocks.append(best_sub_block)
                temp_last = best_sub_block
                
        split_blocks.append(total_blocks)
        
        unique_blocks = sorted(list(set(split_blocks)))
        split_samples = [b * block_size for b in unique_blocks]
        split_samples[-1] = total_samples
        return split_samples, sr

    def _split_audio(self, audio_path, split_samples, sr):
        """
        Ghi các phần chia nhỏ từ file gốc.
        """
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        with sf.SoundFile(audio_path) as f:
            for i in range(len(split_samples) - 1):
                start = split_samples[i]
                end = split_samples[i+1]
                f.seek(start)
                data = f.read(end - start)
                
                chunk_path = os.path.join(self.input_dir, f"{basename}_chunk_{i:03d}.wav")
                sf.write(chunk_path, data, sr)

