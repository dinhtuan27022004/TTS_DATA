from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Callable
import re
import numpy as np


def split_text(
    text: str,
    min_words: int = 10,
) -> List[str]:
    """Tách text đầu vào thành các đoạn nhỏ để tổng hợp âm thanh.

    Chiến lược tách:
    1. Tách theo các dấu câu cuối câu (. ! ? ;).
    2. Gộp các đoạn quá ngắn (< min_words từ) với đoạn kế tiếp.

    Args:
        text: Văn bản đầu vào cần tách.
        min_words: Số từ tối thiểu trong một đoạn (mặc định 10).
            Đoạn ngắn hơn sẽ được gộp vào đoạn kế tiếp.

    Returns:
        Danh sách các chuỗi đoạn text, mỗi phần tử là một đoạn
        để tổng hợp thành audio riêng trước khi ghép lại.
    """
    # Bước 1: Tách sơ bộ theo dấu câu, giữ lại dấu câu cuối đoạn
    raw_chunks = re.split(r'(?<=[.!?;])\s+', text.strip())
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

    # Bước 2: Gộp các chunk quá ngắn vào nhau
    merged: List[str] = []
    buffer = ""
    for chunk in raw_chunks:
        candidate = (buffer + " " + chunk).strip() if buffer else chunk
        word_count = len(candidate.split())
        if word_count < min_words:
            buffer = candidate
        else:
            merged.append(candidate)
            buffer = ""
    if buffer:
        # Gộp phần dư vào đoạn cuối (nếu có) hoặc thêm mới
        if merged:
            merged[-1] = (merged[-1] + " " + buffer).strip()
        else:
            merged.append(buffer)

    return merged if merged else [text.strip()]


def concat_audio(segments: List[np.ndarray], sample_rate: int) -> np.ndarray:
    """Ghép nhiều đoạn audio numpy array thành một array thống nhất.

    Args:
        segments: Danh sách các numpy array audio (float32).
        sample_rate: Sample rate chung của tất cả các đoạn.
            (dùng để thêm khoảng lặng ngắn 50ms giữa các đoạn)

    Returns:
        numpy array float32 chứa toàn bộ audio đã ghép.
    """
    if not segments:
        return np.array([], dtype=np.float32)
    # Thêm khoảng lặng 50ms giữa các đoạn
    silence_samples = int(0.05 * sample_rate)
    silence = np.zeros(silence_samples, dtype=np.float32)
    combined = []
    for i, seg in enumerate(segments):
        seg = np.squeeze(seg).astype(np.float32)
        combined.append(seg)
        if i < len(segments) - 1:
            combined.append(silence)
    return np.concatenate(combined)


class TTSModel(ABC):
    """Abstract base class cho mô hình TTS.

    Mọi mô hình TTS cần đánh giá phải kế thừa class này
    và implement phương thức synthesize.
    """

    @abstractmethod
    def synthesize(
        self,
        gen_text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        split_sentences: bool = False,
        min_words: int = 10,
        custom_duration_fn: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, int]:
        """Tổng hợp audio từ text.

        Args:
            gen_text: Nội dung text cần tổng hợp thành giọng nói.
            ref_audio_path: Đường dẫn đến audio tham chiếu (dùng cho voice cloning).
                Nếu None, model sử dụng ref audio mặc định.
            ref_text: Transcript của ref audio.
                Nếu None, model sử dụng ref_text mặc định hoặc gen_text.
            split_sentences: Nếu True, tách gen_text thành nhiều đoạn nhỏ,
                tổng hợp từng đoạn rồi ghép lại thành 1 file audio.
            min_words: Số từ tối thiểu trong mỗi đoạn tách
                (chỉ dùng khi split_sentences=True). Mặc định 10.

        Returns:
            Tuple gồm:
                - numpy array chứa dữ liệu audio (float32)
                - sample rate (int)
        """
        ...
