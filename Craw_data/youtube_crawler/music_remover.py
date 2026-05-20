"""
Phase 3: Music Removal
Loại bỏ nhạc nền từ audio sử dụng Facebook Demucs (htdemucs).
Chỉ giữ lại vocals để chuẩn bị cho transcription.
"""

import os
import json
import wave
import logging
from typing import List

import torch
import torchaudio
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MusicRemover:
    """
    Loại bỏ nhạc nền từ audio sử dụng Facebook Demucs.
    
    - Load model Demucs (htdemucs)
    - Tách vocals từ mỗi file WAV trong Step_0
    - Lưu vocals vào Step_1 (giữ nguyên tên file)
    - Resume: skip file đã có output trong Step_1
    - Lưu stats.json: total_files, total_duration_seconds, avg_duration_seconds
    """

    def __init__(
        self,
        input_dir: str = "Youtube_Data/Step_0",
        output_dir: str = "Youtube_Data/Step_1",
        model_name: str = "htdemucs"
    ):
        """
        Args:
            input_dir: Thư mục Step_0 chứa WAV gốc
            output_dir: Thư mục Step_1 để lưu vocals
            model_name: Tên model Demucs (default: htdemucs)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.stats_path = os.path.join(output_dir, "stats.json")

        # Model sẽ được load khi cần (lazy loading)
        self.model = None
        self.device = None

    def _load_model(self):
        """
        Load model Demucs (htdemucs).
        Tự động chọn GPU nếu có, fallback sang CPU.
        """
        from demucs.pretrained import get_model

        # Chọn device: GPU nếu có, không thì CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Sử dụng GPU (CUDA) cho Demucs")
        else:
            self.device = torch.device("cpu")
            logger.info("Sử dụng CPU cho Demucs")

        # Load model
        logger.info(f"Đang load model Demucs: {self.model_name}...")
        self.model = get_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model {self.model_name} đã load thành công!")

    def process_all(self) -> List[str]:
        """
        Xử lý tất cả file WAV trong input_dir, resume từ vị trí dừng.
        
        - Liệt kê tất cả file .wav trong input_dir
        - Skip file đã có output trong output_dir (resume)
        - Tách vocals và lưu vào output_dir
        - Lưu stats.json sau khi hoàn thành
        
        Returns:
            Danh sách file đã xử lý thành công
        """
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)

        # Liệt kê tất cả file WAV trong input_dir
        all_wav_files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(".wav")
        ]
        all_wav_files.sort()

        if not all_wav_files:
            logger.warning(f"Không tìm thấy file WAV nào trong {self.input_dir}")
            return []

        logger.info(f"Tìm thấy {len(all_wav_files)} file WAV trong {self.input_dir}")

        # Kiểm tra file nào đã xử lý (resume)
        files_to_process = []
        for filename in all_wav_files:
            output_path = os.path.join(self.output_dir, filename)
            if os.path.exists(output_path):
                logger.info(f"[SKIP] Đã tồn tại: {filename}")
            else:
                files_to_process.append(filename)

        skipped = len(all_wav_files) - len(files_to_process)
        logger.info(f"Đã xử lý trước đó: {skipped} files")
        logger.info(f"Cần xử lý thêm: {len(files_to_process)} files")

        if not files_to_process:
            logger.info("Tất cả file đã được xử lý. Không cần xử lý thêm.")
            self._save_stats()
            return []

        # Load model (chỉ load khi thực sự cần xử lý)
        if self.model is None:
            self._load_model()

        # Xử lý từng file với progress bar
        processed_files = []
        for filename in tqdm(files_to_process, desc="Removing music"):
            input_path = os.path.join(self.input_dir, filename)
            try:
                output_path = self._separate_vocals(input_path)
                processed_files.append(output_path)
                logger.info(f"[OK] {filename}")
            except torch.cuda.OutOfMemoryError:
                # GPU OOM: fallback sang CPU
                logger.warning(f"[OOM] GPU hết bộ nhớ cho {filename}, thử lại với CPU...")
                try:
                    output_path = self._separate_vocals_cpu_fallback(input_path)
                    processed_files.append(output_path)
                    logger.info(f"[OK] {filename} (CPU fallback)")
                except Exception as e:
                    logger.error(f"[FAIL] Không thể xử lý {filename} (CPU fallback): {e}")
            except Exception as e:
                logger.error(f"[FAIL] Không thể xử lý {filename}: {e}")

        # Lưu stats
        self._save_stats()

        logger.info(f"Hoàn thành! Đã xử lý {len(processed_files)} files mới.")
        return processed_files

    def _separate_vocals(self, input_path: str) -> str:
        """
        Tách vocals từ 1 file audio sử dụng Demucs.
        
        Demucs tách audio thành 4 stems: drums, bass, other, vocals.
        Chỉ giữ lại stem "vocals" và lưu vào output_dir.
        
        Args:
            input_path: Đường dẫn file WAV đầu vào
            
        Returns:
            Đường dẫn file vocals WAV đã lưu
        """
        from demucs.apply import apply_model

        # Đọc audio file
        wav, sr = torchaudio.load(input_path)

        # Demucs yêu cầu audio ở sample rate của model
        model_sr = self.model.samplerate
        if sr != model_sr:
            # Resample về sample rate của model
            wav = torchaudio.functional.resample(wav, sr, model_sr)
            sr = model_sr

        # Đảm bảo audio là stereo (2 channels) - Demucs yêu cầu
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)  # Mono -> Stereo
        elif wav.shape[0] > 2:
            wav = wav[:2, :]  # Lấy 2 channels đầu

        # Thêm batch dimension: (channels, samples) -> (1, channels, samples)
        wav = wav.unsqueeze(0).to(self.device)

        # Chạy model Demucs để tách stems
        with torch.no_grad():
            sources = apply_model(self.model, wav, device=self.device)

        # sources shape: (1, num_sources, channels, samples)
        # Tìm index của stem "vocals"
        vocals_idx = self.model.sources.index("vocals")
        vocals = sources[0, vocals_idx]  # (channels, samples)

        # Chuyển về CPU để lưu
        vocals = vocals.cpu()

        # Lưu file vocals (giữ nguyên tên file gốc)
        filename = os.path.basename(input_path)
        output_path = os.path.join(self.output_dir, filename)

        torchaudio.save(output_path, vocals, sr)

        return output_path

    def _separate_vocals_cpu_fallback(self, input_path: str) -> str:
        """
        Fallback: tách vocals trên CPU khi GPU OOM.
        Xử lý theo chunks nếu file quá lớn.
        
        Args:
            input_path: Đường dẫn file WAV đầu vào
            
        Returns:
            Đường dẫn file vocals WAV đã lưu
        """
        from demucs.apply import apply_model

        # Đọc audio file
        wav, sr = torchaudio.load(input_path)

        # Resample nếu cần
        model_sr = self.model.samplerate
        if sr != model_sr:
            wav = torchaudio.functional.resample(wav, sr, model_sr)
            sr = model_sr

        # Đảm bảo stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]

        # Xử lý trên CPU
        cpu_device = torch.device("cpu")
        model_cpu = self.model.to(cpu_device)

        wav_batch = wav.unsqueeze(0)  # (1, channels, samples)

        with torch.no_grad():
            sources = apply_model(model_cpu, wav_batch, device=cpu_device)

        # Đưa model về lại GPU cho lần xử lý tiếp theo
        if self.device.type == "cuda":
            self.model.to(self.device)

        # Lấy vocals
        vocals_idx = self.model.sources.index("vocals")
        vocals = sources[0, vocals_idx]  # (channels, samples)

        # Lưu file
        filename = os.path.basename(input_path)
        output_path = os.path.join(self.output_dir, filename)

        torchaudio.save(output_path, vocals, sr)

        return output_path

    def _save_stats(self):
        """
        Lưu stats.json: total_files, total_duration_seconds, avg_duration_seconds.
        Tính duration từ các file WAV trong output_dir.
        """
        total_files = 0
        total_duration = 0.0

        # Duyệt tất cả file WAV trong output_dir
        for filename in os.listdir(self.output_dir):
            if not filename.lower().endswith(".wav"):
                continue

            filepath = os.path.join(self.output_dir, filename)
            try:
                with wave.open(filepath, "r") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
                    total_duration += duration
                    total_files += 1
            except Exception as e:
                logger.warning(f"Không thể đọc duration của {filename}: {e}")
                total_files += 1

        # Tính trung bình
        avg_duration = total_duration / total_files if total_files > 0 else 0.0

        stats = {
            "total_files": total_files,
            "total_duration_seconds": round(total_duration, 2),
            "avg_duration_seconds": round(avg_duration, 2)
        }

        try:
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"Stats: {total_files} files, {total_duration:.1f}s total, {avg_duration:.1f}s avg")
        except Exception as e:
            logger.error(f"Lỗi lưu stats.json: {e}")
