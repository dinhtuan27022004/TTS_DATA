"""
Whisper Large v3 ASR component.

Wrapper này dùng faster-whisper để transcribe tiếng Việt và trả về cùng
TranscriptionResult mà pipeline YouTube crawler đang dùng.
"""

import logging
from typing import Optional
import soundfile as sf
from faster_whisper import WhisperModel

from Craw_data.youtube_crawler.models import TranscriptionResult, WordTimestamp
from .base import ASRModel

logger = logging.getLogger(__name__)


class WhisperLargeV3ASR(ASRModel):
    """ASR component sử dụng faster-whisper Large v3."""

    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "vi",
        device: Optional[str] = "cuda",
        compute_type: Optional[str] = "float16",
        beam_size: int = 5,
        num_workers: int = 1,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        use_batched_pipeline: bool = False,
    ):
        self.model_name = model_name
        self.language = language
        self.device = device if device else "cuda"
        self.compute_type = compute_type if compute_type else "float16"
        self.beam_size = beam_size
        self.num_workers = num_workers
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter
        self.use_batched_pipeline = use_batched_pipeline
        self.model: Optional[WhisperModel] = None
        self.batched_model = None

    def _load_model(self):
        if self.model is not None:
            return

        logger.debug(
            "Chuẩn bị Whisper model=%s device=%s compute_type=%s",
            self.model_name,
            self.device,
            self.compute_type,
        )
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=self.num_workers,
        )

        if self.use_batched_pipeline:
            from faster_whisper import BatchedInferencePipeline
            logger.info("Khởi tạo BatchedInferencePipeline cho Whisper...")
            self.batched_model = BatchedInferencePipeline(model=self.model)

    def transcribe(self, wav_path: str) -> TranscriptionResult:
        self._load_model()
        
        info = sf.info(wav_path)
        duration = float(info.duration)
        
        if self.use_batched_pipeline and self.batched_model is not None:
            segments_gen, _ = self.batched_model.transcribe(
                wav_path,
                batch_size=16,
                language=self.language,
                beam_size=self.beam_size,
                word_timestamps=self.word_timestamps,
                condition_on_previous_text=False,
            )
        else:
            segments_gen, _ = self.model.transcribe(
                wav_path,
                language=self.language,
                beam_size=self.beam_size,
                word_timestamps=self.word_timestamps,
                vad_filter=self.vad_filter,
                condition_on_previous_text=False,
            )
        
        word_timestamps = []
        full_text_parts = []
        
        for segment in segments_gen:
            text = segment.text.strip()
            if text:
                full_text_parts.append(text)
            
            if self.word_timestamps and segment.words:
                for word_info in segment.words:
                    word = word_info.word.strip()
                    if not word:
                        continue
                    start = max(0.0, float(word_info.start))
                    end = min(duration, float(word_info.end))
                    if end <= start:
                        end = min(duration, start + 0.05)
                        
                    word_timestamps.append(
                        WordTimestamp(
                            word=word,
                            start_time=round(start, 3),
                            end_time=round(end, 3),
                        )
                    )
                    
        full_text = " ".join(full_text_parts)
        
        return TranscriptionResult(
            wav_path=wav_path,
            full_text=full_text,
            word_timestamps=word_timestamps,
            duration=duration,
        )

    def load(self):
        self._load_model()

    def transcribe_text(self, wav_path: str, batch_size: Optional[int] = None) -> str:
        if batch_size is not None and self.use_batched_pipeline:
            # We can run with batched pipeline if enabled
            res = self.transcribe(wav_path)
            return res.full_text
        res = self.transcribe(wav_path)
        return res.full_text

