"""
Data models cho hệ thống đánh giá TTS.

Chứa các dataclass mô tả cấu trúc dữ liệu chính:
- EvalSample: thông tin một mẫu đánh giá
- MetricResult: kết quả tính toán metric cho một mẫu
- EvaluationReport: toàn bộ kết quả đánh giá
- RunHistoryEntry: thông tin lần chạy đánh giá
- ResultFileData: dữ liệu file kết quả JSON
- TTSModel: abstract base class cho mô hình TTS
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EvalSample:
    """Chứa thông tin một mẫu đánh giá.

    Attributes:
        sample_id: ID duy nhất của mẫu
        text: Nội dung text cần tổng hợp
        audio_path: Đường dẫn đến file audio ground truth
    """

    sample_id: str
    text: str
    audio_path: str


@dataclass
class MetricResult:
    """Chứa kết quả tính toán metric cho một mẫu.

    Attributes:
        sample_id: ID của mẫu đã đánh giá
        text: Nội dung text gốc
        mcd: Mel Cepstral Distortion (None nếu lỗi)
        pesq: PESQ score (None nếu lỗi)
        stoi: STOI score trong [0, 1] (None nếu lỗi)
        utmos: UTMOS MOS prediction trong [1.0, 5.0] (None nếu lỗi)
        f0_correlation: F0 correlation trong [-1.0, 1.0] (None nếu lỗi)
        wer: Word Error Rate >= 0 (None nếu lỗi)
    """

    sample_id: str
    text: str
    mcd: Optional[float] = None
    pesq: Optional[float] = None
    stoi: Optional[float] = None
    utmos: Optional[float] = None
    f0_correlation: Optional[float] = None
    wer: Optional[float] = None
    cer: Optional[float] = None
    transcription: Optional[str] = None


@dataclass
class EvaluationReport:
    """Chứa toàn bộ kết quả đánh giá.

    Attributes:
        results: Danh sách MetricResult cho từng mẫu
        chart_paths: Danh sách đường dẫn đến các file biểu đồ đã tạo
        summary_statistics: Thống kê tổng hợp (mean, std, min, max) cho từng metric
        csv_path: Đường dẫn đến file CSV kết quả
    """

    results: List[MetricResult]
    chart_paths: List[str] = field(default_factory=list)
    summary_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    csv_path: Optional[str] = None


@dataclass
class RunHistoryEntry:
    """Thông tin một lần chạy đánh giá.

    Attributes:
        model_name: Tên mô hình TTS đã đánh giá
        metric_name: Tên metric đã tính toán
        start_time: Thời gian bắt đầu (ISO format string)
        end_time: Thời gian kết thúc (ISO format string)
        num_samples: Số mẫu đã xử lý
        status: Trạng thái hoàn thành ("completed", "failed", "partial")
    """

    model_name: str
    metric_name: str
    start_time: str
    end_time: str
    num_samples: int
    status: str


@dataclass
class ResultFileData:
    """Dữ liệu file kết quả JSON cho một model và metric cụ thể.

    Attributes:
        model_name: Tên mô hình TTS
        metric_name: Tên metric
        samples: Danh sách kết quả từng mẫu (sample_id, value, text)
        summary: Thống kê tổng hợp (mean, std, min, max)
    """

    model_name: str
    metric_name: str
    samples: List[Dict[str, object]] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)


class TTSModel(ABC):
    """Abstract base class cho mô hình TTS.

    Mọi mô hình TTS cần đánh giá phải kế thừa class này
    và implement phương thức synthesize.
    """

    @abstractmethod
    def synthesize(self, gen_text: str, ref_audio_path: Optional[str] = None, ref_text: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """Tổng hợp audio từ text.

        Args:
            gen_text: Nội dung text cần tổng hợp thành giọng nói.
            ref_audio_path: Đường dẫn đến audio tham chiếu (dùng cho voice cloning).
                Nếu None, model sử dụng ref audio mặc định.
            ref_text: Transcript của ref audio.
                Nếu None, model sử dụng ref_text mặc định hoặc gen_text.

        Returns:
            Tuple gồm:
                - numpy array chứa dữ liệu audio (float32)
                - sample rate (int)
        """
        ...
