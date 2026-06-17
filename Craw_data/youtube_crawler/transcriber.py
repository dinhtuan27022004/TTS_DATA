"""
Phase 4: Transcription
Transcribe audio tiếng Việt bằng Whisper Large v3 (thông qua faster-whisper).
"""

import os
import json
import logging
from typing import List, Optional
import torch
from tqdm import tqdm
from dataclasses import asdict

from .models import TranscriptionResult

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Transcriber:
    """
    Transcribe audio tiếng Việt bằng Whisper Large v3.
    
    - Transcribe từng file WAV trong Step_1
    - Trích xuất word-level timestamps
    - Lưu timestamps dạng JSON (cùng tên file .json)
    - Resume: skip file đã có .json tương ứng
    """

    def __init__(
        self,
        input_dir: str = "Youtube_Data/Step_1",
        model_name: str = "large-v3",
        language: str = "vi",
        device: Optional[str] = None,
        compute_type: str = "int8",
        beam_size: int = 5,
        num_workers: int = 1,
    ):
        """
        Args:
            input_dir: Thư mục Step_1 chứa vocals WAV
            model_name: Tên model Whisper. Dùng "large-v3"
            language: Mã ngôn ngữ cho Whisper
            device: Device cho Whisper ("cuda", "cpu", hoặc None = auto)
            compute_type: compute_type cho faster-whisper (None = auto)
            beam_size: Beam size cho Whisper
            num_workers: Số worker nội bộ của faster-whisper
        """
        self.input_dir = input_dir
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.num_workers = num_workers

        # Model sẽ được load khi cần (lazy loading)
        self.model = None

    def _load_model(self):
        """Load model Whisper Large v3 qua ASR component."""
        from components.asr import WhisperLargeV3ASR

        logger.info(f"Đang load Whisper ASR component: {self.model_name}...")
        self.model = WhisperLargeV3ASR(
            model_name=self.model_name,
            language=self.language,
            device=self.device,
            compute_type=self.compute_type,
            beam_size=self.beam_size,
            num_workers=self.num_workers,
            word_timestamps=True,
            use_batched_pipeline=True,  # Kích hoạt BatchedInferencePipeline
        )
        self.model._load_model()
        logger.info(f"Whisper ASR component {self.model_name} đã load thành công!")

    def transcribe_all(self) -> List[TranscriptionResult]:
        import time
        results = []
        max_empty_retries = 3
        empty_retries = 0

        # Load model trước vòng lặp nếu chưa load
        if self.model is None:
            self._load_model()

        while True:
            # Quét trực tiếp thư mục Step_1
            demucs_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".wav")]
            demucs_files.sort()

            # Kiểm tra file nào đã có .json (resume)
            files_to_process = []
            for filename in demucs_files:
                json_path = os.path.join(
                    self.input_dir,
                    os.path.splitext(filename)[0] + ".json"
                )
                if os.path.exists(json_path):
                    continue
                
                wav_path = os.path.join(self.input_dir, filename)
                if os.path.exists(wav_path):
                    files_to_process.append(filename)

            if not files_to_process:
                empty_retries += 1
                if empty_retries > max_empty_retries:
                    logger.info("Không có file mới nào sau nhiều lần thử. Kết thúc Phase 4.")
                    break
                logger.info(f"Chưa có file mới. Chờ 60 giây và thử lại... ({empty_retries}/{max_empty_retries})")
                time.sleep(60)
                continue

            empty_retries = 0
            logger.info(f"Cần transcribe thêm lô mới: {len(files_to_process)} files")

            for filename in tqdm(files_to_process, desc="Transcribing"):
                wav_path = os.path.join(self.input_dir, filename)
                try:
                    result = self._transcribe_single(wav_path)
                    if result is not None:
                        self._save_json(result)
                        results.append(result)
                        logger.info(f"[OK] {filename} - '{result.full_text[:50]}...'")
                    else:
                        empty_result = TranscriptionResult(wav_path=wav_path, full_text="", word_timestamps=[], duration=0.0)
                        self._save_json(empty_result)
                        logger.warning(f"[SKIP] Không có speech: {filename}")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error(f"[CRITICAL] Hết VRAM (OOM) khi xử lý {filename}. Chờ 5 phút rồi thử lại...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        time.sleep(300)
                        break
                    else:
                        logger.error(f"[FAIL] Lỗi transcribe {filename}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"[FAIL] Lỗi transcribe {filename}: {e}", exc_info=True)

        logger.info(f"Hoàn thành! Đã transcribe tổng cộng {len(results)} files mới trong phiên này.")
        return results

    def _transcribe_single(self, wav_path: str) -> Optional[TranscriptionResult]:
        """
        Transcribe 1 file WAV bằng Whisper, trả về text + word-level timestamps.
        """
        result = self.model.transcribe(wav_path)
        if not result.full_text.strip():
            return None
        return result

    def _save_json(self, result: TranscriptionResult):
        """Lưu kết quả transcribe ra JSON cùng tên với WAV."""
        json_path = os.path.splitext(result.wav_path)[0] + ".json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu JSON cho {result.wav_path}: {e}")
