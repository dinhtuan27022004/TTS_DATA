"""
Pydantic schemas cho F5-TTS API.

Định nghĩa các request/response model được dùng trong FastAPI endpoints.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Trạng thái của một TTS job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TTSRequest(BaseModel):
    """Request body cho POST /api/tts."""
    model_name: str = Field(..., description="Tên model checkpoint, vd: f5-tts-70000")
    ref_audio_b64: str = Field(..., description="Base64-encoded WAV bytes của reference audio")
    ref_text: str = Field(..., description="Transcript tương ứng với reference audio")
    target_text: str = Field(..., description="Nội dung text cần tổng hợp thành giọng nói")
    split_sentences: bool = Field(
        False,
        description="Nếu True, tách target_text thành nhiều câu rồi ghép audio lại.",
    )
    min_words: int = Field(
        10,
        ge=1,
        description="Số từ tối thiểu trong mỗi câu khi split_sentences=True.",
    )
    nfe_step: int = Field(
        64,
        ge=16,
        le=128,
        description="Số bước chạy DiT (NFE Steps) cho ODE solver.",
    )


class TTSSubmitResponse(BaseModel):
    """Response cho POST /api/tts - trả về job_id ngay lập tức."""
    job_id: str = Field(..., description="UUID định danh job")


class TTSResultResponse(BaseModel):
    """Response cho GET /api/tts/{job_id} - trạng thái và kết quả."""
    status: JobStatus
    audio_url: Optional[str] = Field(None, description="URL tải file WAV kết quả")
    duration: Optional[float] = Field(None, description="Thời lượng audio (giây)")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu job thất bại")


class ModelInfo(BaseModel):
    """Thông tin của một model checkpoint."""
    name: str
    model_path: str
    vocab_path: str


class ModelsListResponse(BaseModel):
    """Response cho GET /api/models."""
    models: list[ModelInfo]


class SampleInfo(BaseModel):
    """Thông tin của một sample tham chiếu."""
    name: str
    audio_path: str
    text_content: str
    wps: Optional[float] = None


class SamplesListResponse(BaseModel):
    """Response cho GET /api/samples."""
    samples: list[SampleInfo]


class CompareSubmitResponse(BaseModel):
    """Response cho POST /api/tts/compare - trả về job_id của cả 2 model."""
    job_id_v0: str = Field(..., description="UUID của job chạy model f5-tts-v0")
    job_id_selected: str = Field(..., description="UUID của job chạy model đang chọn")
    model_v0: str = Field(..., description="Tên model v0 (baseline)")
    model_selected: str = Field(..., description="Tên model đang so sánh")


class CompareJobResult(BaseModel):
    """Kết quả của một trong hai model trong so sánh."""
    model_name: str
    status: JobStatus
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None


class CompareResultResponse(BaseModel):
    """Response tổng hợp kết quả so sánh 2 model."""
    v0: CompareJobResult
    selected: CompareJobResult
