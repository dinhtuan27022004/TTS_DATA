from abc import ABC, abstractmethod

from Craw_data.youtube_crawler.models import TranscriptionResult


class ASRModel(ABC):
    """Abstract base class cho mô hình ASR."""

    @abstractmethod
    def transcribe(self, wav_path: str) -> TranscriptionResult:
        """Nhận dạng giọng nói từ một file WAV."""
        ...
