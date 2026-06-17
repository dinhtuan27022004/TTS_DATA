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
    CompareJobResult,
    CompareResultResponse,
    CompareSubmitResponse,
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


# ─── Tên cố định của model baseline ─────────────────────────────────────────
BASELINE_MODEL = "f5-tts-v0"


# ─── Models endpoint ──────────────────────────────────────────────────────────
@app.get("/api/models", response_model=ModelsListResponse, tags=["Discovery"])
async def list_models() -> ModelsListResponse:
    """Trả về danh sách checkpoint models (loại trừ baseline v0 dùng để so sánh)."""
    models = [m for m in scan_models() if m.name != BASELINE_MODEL]
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
            wps=s.get("wps"),
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
        split_sentences=request.split_sentences,
        min_words=request.min_words,
        nfe_step=request.nfe_step,
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


# ─── Compare Submit ───────────────────────────────────────────────────────────
@app.post("/api/tts/compare", response_model=CompareSubmitResponse, tags=["Compare"])
async def submit_compare(request: TTSRequest) -> CompareSubmitResponse:
    """Submit 2 TTS job song song: baseline v0 và model đang chọn.

    Body:
        model_name    – Tên model muốn so sánh với v0
        ref_audio_b64 – Base64 của file WAV tham chiếu
        ref_text      – Transcript tương ứng
        target_text   – Text cần đọc

    Returns:
        { "job_id_v0": "...", "job_id_selected": "...",
          "model_v0": "f5-tts-v0", "model_selected": "..." }
    """
    all_models = {m.name for m in scan_models()}

    # Kiểm tra model đang chọn
    if request.model_name not in all_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' không tồn tại. "
                   f"Các model có sẵn: {sorted(all_models)}",
        )

    # Kiểm tra baseline v0 tồn tại
    if BASELINE_MODEL not in all_models:
        raise HTTPException(
            status_code=500,
            detail=f"Baseline model '{BASELINE_MODEL}' không tồn tại trên server.",
        )

    if not request.target_text.strip():
        raise HTTPException(status_code=400, detail="target_text không được để trống.")
    if not request.ref_text.strip():
        raise HTTPException(status_code=400, detail="ref_text không được để trống.")
    if not request.ref_audio_b64.strip():
        raise HTTPException(status_code=400, detail="ref_audio_b64 không được để trống.")

    # Submit 2 job tuần tự vào queue (worker pool chỉ 1 worker nên sẽ chạy lần lượt)
    job_id_v0 = queue_manager.submit_job(
        model_name=BASELINE_MODEL,
        ref_audio_b64=request.ref_audio_b64,
        ref_text=request.ref_text,
        target_text=request.target_text,
        split_sentences=request.split_sentences,
        min_words=request.min_words,
        nfe_step=request.nfe_step,
    )
    job_id_selected = queue_manager.submit_job(
        model_name=request.model_name,
        ref_audio_b64=request.ref_audio_b64,
        ref_text=request.ref_text,
        target_text=request.target_text,
        split_sentences=request.split_sentences,
        min_words=request.min_words,
        nfe_step=request.nfe_step,
    )

    logger.info(
        "Compare jobs submitted: v0=%s, selected=%s (model=%s)",
        job_id_v0, job_id_selected, request.model_name,
    )
    return CompareSubmitResponse(
        job_id_v0=job_id_v0,
        job_id_selected=job_id_selected,
        model_v0=BASELINE_MODEL,
        model_selected=request.model_name,
    )


# ─── Compare Result ────────────────────────────────────────────────────────────
@app.get("/api/tts/compare/result", response_model=CompareResultResponse, tags=["Compare"])
async def get_compare_result(
    job_id_v0: str,
    job_id_selected: str,
    model_v0: str = BASELINE_MODEL,
    model_selected: str = "",
) -> CompareResultResponse:
    """Lấy kết quả tổng hợp của cả 2 job so sánh.

    Query params:
        job_id_v0       – job_id của model v0
        job_id_selected – job_id của model đang so sánh
        model_v0        – tên model v0 (mặc định: f5-tts-v0)
        model_selected  – tên model đang so sánh
    """
    def _make_job_result(job_id: str, model_name: str) -> CompareJobResult:
        result = queue_manager.get_result(job_id)
        if result is None:
            return CompareJobResult(
                model_name=model_name,
                status=JobStatus.FAILED,
                error=f"Không tìm thấy job: {job_id}",
            )
        audio_url = None
        if result.status == JobStatus.COMPLETED and result.audio_path:
            filename = os.path.basename(result.audio_path)
            audio_url = f"/outputs/{filename}"
        return CompareJobResult(
            model_name=model_name,
            status=result.status,
            audio_url=audio_url,
            duration=result.duration,
            error=result.error,
        )

    return CompareResultResponse(
        v0=_make_job_result(job_id_v0, model_v0),
        selected=_make_job_result(job_id_selected, model_selected),
    )


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """Kiểm tra server còn sống."""
    return {"status": "ok"}
