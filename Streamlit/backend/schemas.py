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


class SamplesListResponse(BaseModel):
    """Response cho GET /api/samples."""
    samples: list[SampleInfo]
