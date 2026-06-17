"""
Queue Manager cho F5-TTS.

Xử lý TTS job bất đồng bộ bằng ThreadPoolExecutor + queue.Queue.
Worker chạy ngầm, nhận job, gọi model, lưu kết quả vào job_store.
"""

import base64
import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf

from .schemas import JobStatus

logger = logging.getLogger(__name__)

# ─── Đường dẫn ───────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(_THIS_DIR)  # /home/reg/TTS_DATA
OUTPUTS_DIR = os.path.join(_THIS_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


@dataclass
class TTSJob:
    """Dữ liệu đầy đủ của một TTS job trong hàng đợi."""

    job_id: str
    model_name: str
    ref_audio_b64: str          # Base64-encoded WAV bytes
    ref_text: str
    target_text: str
    split_sentences: bool = False
    min_words: int = 10
    nfe_step: int = 64
    created_at: float = field(default_factory=time.time)


@dataclass
class JobResult:
    """Kết quả sau khi xử lý một TTS job."""

    job_id: str
    status: JobStatus
    audio_path: Optional[str] = None    # Đường dẫn file WAV trên disk
    duration: Optional[float] = None    # Thời lượng audio (giây)
    error: Optional[str] = None


class QueueManager:
    """Singleton quản lý hàng đợi và worker pool cho TTS jobs.

    Architecture:
        POST /api/tts
            → submit_job() → job_queue.put()
            → trả về job_id ngay lập tức

        Worker thread:
            → lấy job từ queue
            → gọi model.synthesize()
            → lưu file WAV vào outputs/
            → cập nhật job_store
    """

    def __init__(self, num_workers: int = 1) -> None:
        # job_id → JobResult (thread-safe nhờ RLock)
        self._job_store: dict[str, JobResult] = {}
        self._store_lock = threading.RLock()

        # Hàng đợi chứa TTSJob
        self._job_queue: queue.Queue[TTSJob] = queue.Queue()

        # ThreadPoolExecutor để chạy worker
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="tts-worker"
        )

        # Khởi động dispatcher thread
        self._stop_event = threading.Event()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="tts-dispatcher"
        )
        self._dispatcher.start()
        logger.info("QueueManager khởi động với %d worker(s).", num_workers)

    # ── Public API ────────────────────────────────────────────────────────────

    def submit_job(
        self,
        model_name: str,
        ref_audio_b64: str,
        ref_text: str,
        target_text: str,
        split_sentences: bool = False,
        min_words: int = 10,
        nfe_step: int = 64,
    ) -> str:
        """Tạo job mới và đưa vào hàng đợi.

        Args:
            model_name: Tên checkpoint model.
            ref_audio_b64: Reference audio dưới dạng base64.
            ref_text: Transcript của reference audio.
            target_text: Text cần tổng hợp.
            split_sentences: Tách text thành nhiều câu.
            min_words: Số từ tối thiểu trong mỗi câu.
            nfe_step: Số bước chạy DiT.

        Returns:
            job_id dạng UUID string.
        """
        job_id = str(uuid.uuid4())
        job = TTSJob(
            job_id=job_id,
            model_name=model_name,
            ref_audio_b64=ref_audio_b64,
            ref_text=ref_text,
            target_text=target_text,
            split_sentences=split_sentences,
            min_words=min_words,
            nfe_step=nfe_step,
        )

        # Đăng ký trạng thái ban đầu
        with self._store_lock:
            self._job_store[job_id] = JobResult(
                job_id=job_id, status=JobStatus.PENDING
            )

        self._job_queue.put(job)
        logger.info("Đã submit job %s (model=%s, split=%s)", job_id, model_name, split_sentences)
        return job_id

    def get_result(self, job_id: str) -> Optional[JobResult]:
        """Trả về JobResult hiện tại của job, hoặc None nếu không tồn tại.

        Args:
            job_id: UUID của job cần kiểm tra.

        Returns:
            JobResult hoặc None.
        """
        with self._store_lock:
            return self._job_store.get(job_id)

    def shutdown(self) -> None:
        """Dừng worker pool và dispatcher thread."""
        logger.info("Shutting down QueueManager...")
        self._stop_event.set()
        self._executor.shutdown(wait=False)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _dispatch_loop(self) -> None:
        """Vòng lặp lấy job từ queue và gửi vào executor."""
        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=1.0)
                self._executor.submit(self._process_job, job)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("Lỗi dispatcher: %s", exc)

    def _process_job(self, job: TTSJob) -> None:
        """Worker function: xử lý một TTSJob từ đầu đến cuối.

        Luồng:
            1. Cập nhật status → PROCESSING
            2. Decode base64 audio → lưu temp WAV
            3. Load model qua model_manager
            4. Gọi synthesize()
            5. Lưu kết quả WAV vào outputs/
            6. Cập nhật status → COMPLETED / FAILED
        """
        logger.info("Bắt đầu xử lý job %s", job.job_id)
        self._update_status(job.job_id, JobStatus.PROCESSING)

        ref_tmp_path: Optional[str] = None
        try:
            # ── 1. Decode reference audio ──────────────────────────────────
            ref_audio_bytes = base64.b64decode(job.ref_audio_b64)
            ref_tmp_path = os.path.join(
                OUTPUTS_DIR, f"ref_tmp_{job.job_id}.wav"
            )
            with open(ref_tmp_path, "wb") as fh:
                fh.write(ref_audio_bytes)

            # ── 2. Load model (có cache) ───────────────────────────────────
            from .model_manager import model_manager, scan_models

            models = {m.name: m for m in scan_models()}
            if job.model_name not in models:
                raise ValueError(f"Model '{job.model_name}' không tồn tại.")

            model = model_manager.get_model(models[job.model_name])

            # ── 3. Synthesize ───────────────────────────────────────────────────────
            audio_array, sample_rate = model.synthesize(
                gen_text=job.target_text,
                ref_audio_path=ref_tmp_path,
                ref_text=job.ref_text,
                split_sentences=job.split_sentences,
                min_words=job.min_words,
                nfe_step=job.nfe_step,
            )

            # ── 4. Lưu file kết quả ───────────────────────────────────────
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_filename = f"{timestamp}.wav"
            out_path = os.path.join(OUTPUTS_DIR, out_filename)
            # Tránh trùng tên nếu nhiều job chạy/hoàn thành cùng giây (ví dụ: chế độ so sánh 2 model)
            if os.path.exists(out_path):
                out_filename = f"{timestamp}_{job.job_id[:8]}.wav"
                out_path = os.path.join(OUTPUTS_DIR, out_filename)
            sf.write(out_path, audio_array, sample_rate)

            duration = len(audio_array) / sample_rate
            logger.info(
                "Job %s hoàn thành: %s (%.2f giây)", job.job_id, out_path, duration
            )

            with self._store_lock:
                self._job_store[job.job_id] = JobResult(
                    job_id=job.job_id,
                    status=JobStatus.COMPLETED,
                    audio_path=out_path,
                    duration=duration,
                )

        except Exception as exc:
            logger.exception("Job %s thất bại: %s", job.job_id, exc)
            with self._store_lock:
                self._job_store[job.job_id] = JobResult(
                    job_id=job.job_id,
                    status=JobStatus.FAILED,
                    error=str(exc),
                )
        finally:
            # Xoá file temp
            if ref_tmp_path and os.path.exists(ref_tmp_path):
                try:
                    os.remove(ref_tmp_path)
                except OSError:
                    pass

    def _update_status(self, job_id: str, status: JobStatus) -> None:
        with self._store_lock:
            if job_id in self._job_store:
                self._job_store[job_id].status = status


# ── Module-level singleton ────────────────────────────────────────────────────
queue_manager = QueueManager(num_workers=1)
