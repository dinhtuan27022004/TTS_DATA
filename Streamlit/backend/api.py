"""
FastAPI backend cho F5-TTS Demo.

Endpoint:
    POST /api/tts          – Submit một TTS job
    GET  /api/tts/{job_id} – Polling kết quả job
    GET  /api/models       – Liệt kê checkpoint models
    GET  /api/samples      – Liệt kê reference samples
    GET  /outputs/{filename} – Phục vụ file WAV kết quả (StaticFiles mount)

Chạy:
    cd /home/reg/TTS_DATA/Streamlit
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model_manager import scan_models, scan_samples
from .queue_manager import queue_manager
from .schemas import (
    JobStatus,
    ModelInfo,
    ModelsListResponse,
    SampleInfo,
    SamplesListResponse,
    TTSRequest,
    TTSResultResponse,
    TTSSubmitResponse,
)

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Đường dẫn outputs ───────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(_THIS_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("F5-TTS API server khởi động.")
    yield
    logger.info("F5-TTS API server tắt, dọn dẹp resources...")
    queue_manager.shutdown()


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="F5-TTS Demo API",
    description="API tổng hợp giọng nói tiếng Việt dùng F5-TTS checkpoint.",
    version="1.0.0",
    lifespan=lifespan,
)

# Cho phép Streamlit (chạy cổng khác) gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ file WAV tĩnh qua /outputs/<filename>
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# ─── Models endpoint ──────────────────────────────────────────────────────────
@app.get("/api/models", response_model=ModelsListResponse, tags=["Discovery"])
async def list_models() -> ModelsListResponse:
    """Trả về danh sách tất cả checkpoint models đang có."""
    models = scan_models()
    return ModelsListResponse(models=models)


# ─── Samples endpoint ─────────────────────────────────────────────────────────
@app.get("/api/samples", response_model=SamplesListResponse, tags=["Discovery"])
async def list_samples() -> SamplesListResponse:
    """Trả về danh sách reference samples (cặp wav+txt)."""
    raw = scan_samples()
    samples = [
        SampleInfo(
            name=s["name"],
            audio_path=s["audio_path"],
            text_content=s["text_content"],
        )
        for s in raw
    ]
    return SamplesListResponse(samples=samples)


# ─── TTS Submit ───────────────────────────────────────────────────────────────
@app.post("/api/tts", response_model=TTSSubmitResponse, tags=["TTS"])
async def submit_tts(request: TTSRequest) -> TTSSubmitResponse:
    """Nhận request TTS, đưa vào hàng đợi và trả về job_id ngay.

    Body:
        model_name    – Tên checkpoint (vd: f5-tts-70000)
        ref_audio_b64 – Base64 của file WAV tham chiếu
        ref_text      – Transcript tương ứng
        target_text   – Text cần đọc

    Returns:
        { "job_id": "<uuid>" }
    """
    # Validate model tồn tại
    available = {m.name for m in scan_models()}
    if request.model_name not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' không tồn tại. "
                   f"Các model có sẵn: {sorted(available)}",
        )

    if not request.target_text.strip():
        raise HTTPException(status_code=400, detail="target_text không được để trống.")
    if not request.ref_text.strip():
        raise HTTPException(status_code=400, detail="ref_text không được để trống.")
    if not request.ref_audio_b64.strip():
        raise HTTPException(status_code=400, detail="ref_audio_b64 không được để trống.")

    job_id = queue_manager.submit_job(
        model_name=request.model_name,
        ref_audio_b64=request.ref_audio_b64,
        ref_text=request.ref_text,
        target_text=request.target_text,
    )
    logger.info("Job submitted: %s (model=%s)", job_id, request.model_name)
    return TTSSubmitResponse(job_id=job_id)


# ─── TTS Polling ──────────────────────────────────────────────────────────────
@app.get("/api/tts/{job_id}", response_model=TTSResultResponse, tags=["TTS"])
async def get_tts_result(job_id: str) -> TTSResultResponse:
    """Kiểm tra trạng thái và lấy kết quả của job.

    Returns (khi đang xử lý):
        { "status": "processing" }

    Returns (khi hoàn thành):
        { "status": "completed", "audio_url": "/outputs/xxx.wav", "duration": 12.4 }

    Returns (khi lỗi):
        { "status": "failed", "error": "..." }
    """
    result = queue_manager.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy job: {job_id}")

    audio_url: str | None = None
    if result.status == JobStatus.COMPLETED and result.audio_path:
        filename = os.path.basename(result.audio_path)
        audio_url = f"/outputs/{filename}"

    return TTSResultResponse(
        status=result.status,
        audio_url=audio_url,
        duration=result.duration,
        error=result.error,
    )


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """Kiểm tra server còn sống."""
    return {"status": "ok"}
