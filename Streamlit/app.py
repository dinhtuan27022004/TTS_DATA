"""
F5-TTS Demo — Streamlit Frontend.

Giao diện cho phép:
  1. Chọn checkpoint model F5-TTS.
  2. Chọn sample tham chiếu hoặc upload sample tùy chỉnh.
  3. Nhập text cần synthesize.
  4. Gửi request qua API → polling → phát audio.

Chạy:
    cd /home/reg/TTS_DATA/Streamlit
    streamlit run app.py
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Optional

import requests
import streamlit as st

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL_SEC = 2       # Polling mỗi 2 giây
MAX_POLL_ATTEMPTS = 150     # Tối đa ~5 phút

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F5-TTS Vietnamese Demo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _api_get(path: str, **kwargs) -> dict:
    """Gọi GET request tới backend API."""
    url = f"{API_BASE_URL}{path}"
    resp = requests.get(url, timeout=10, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _api_post(path: str, json: dict) -> dict:
    """Gọi POST request tới backend API."""
    url = f"{API_BASE_URL}{path}"
    resp = requests.post(url, json=json, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch_models() -> list[dict]:
    """Lấy danh sách models từ API. Trả về [] nếu lỗi."""
    try:
        data = _api_get("/api/models")
        return data.get("models", [])
    except Exception as exc:
        logger.warning("Không lấy được danh sách models: %s", exc)
        return []


def _fetch_samples() -> list[dict]:
    """Lấy danh sách samples từ API. Trả về [] nếu lỗi."""
    try:
        data = _api_get("/api/samples")
        return data.get("samples", [])
    except Exception as exc:
        logger.warning("Không lấy được danh sách samples: %s", exc)
        return []


def _encode_audio_b64(audio_bytes: bytes) -> str:
    """Encode bytes thành base64 string."""
    return base64.b64encode(audio_bytes).decode("utf-8")


def _check_api_health() -> bool:
    """Kiểm tra backend API đang chạy không."""
    try:
        _api_get("/health")
        return True
    except Exception:
        return False


def _poll_job(job_id: str) -> dict:
    """Polling job cho đến khi hoàn thành hoặc lỗi.

    Args:
        job_id: UUID của job cần theo dõi.

    Returns:
        JSON result dict cuối cùng từ API.
    """
    progress_bar = st.progress(0, text="Đang khởi tạo job...")
    status_placeholder = st.empty()

    for attempt in range(MAX_POLL_ATTEMPTS):
        try:
            result = _api_get(f"/api/tts/{job_id}")
        except Exception as exc:
            status_placeholder.error(f"Lỗi khi polling: {exc}")
            break

        status = result.get("status", "unknown")
        progress_frac = min((attempt + 1) / MAX_POLL_ATTEMPTS, 0.95)

        if status == "pending":
            progress_bar.progress(progress_frac, text="Đang chờ trong hàng đợi...")
            status_placeholder.info("Trạng thái: **Pending** — đang chờ worker...")
        elif status == "processing":
            progress_bar.progress(progress_frac, text="Đang tổng hợp giọng nói...")
            status_placeholder.info("Trạng thái: **Processing** — mô hình đang tổng hợp...")
        elif status == "completed":
            progress_bar.progress(1.0, text="Hoàn thành!")
            status_placeholder.success("Trạng thái: **Completed**")
            return result
        elif status == "failed":
            progress_bar.progress(1.0, text="Thất bại")
            status_placeholder.error(
                f"Trạng thái: **Failed** — {result.get('error', 'Unknown error')}"
            )
            return result

        time.sleep(POLL_INTERVAL_SEC)

    status_placeholder.error("Timeout: job chạy quá lâu.")
    return {"status": "failed", "error": "Polling timeout"}


def _display_audio_result(result: dict) -> None:
    """Hiển thị audio player và nút download sau khi job hoàn thành."""
    audio_url = result.get("audio_url")
    duration = result.get("duration")

    if not audio_url:
        st.warning("Không có đường dẫn audio trong kết quả.")
        return

    full_url = f"{API_BASE_URL}{audio_url}"
    try:
        audio_resp = requests.get(full_url, timeout=30)
        audio_resp.raise_for_status()
        audio_bytes = audio_resp.content
    except Exception as exc:
        st.error(f"Không tải được file audio: {exc}")
        return

    st.divider()
    st.subheader("🎵 Kết quả tổng hợp")

    if duration:
        st.metric("Thời lượng audio", f"{duration:.2f}s")

    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        label="⬇️ Tải file WAV",
        data=audio_bytes,
        file_name=os.path.basename(audio_url),
        mime="audio/wav",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar — Model & Sample selector
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Cấu hình")
    st.caption("Chọn model và reference sample")
    st.divider()

    # ── API status ────────────────────────────────────────────────────────────
    api_ok = _check_api_health()
    if api_ok:
        st.success("Backend API đang chạy")
    else:
        st.error(
            "Backend API chưa khởi động!\n\n"
            "Chạy:\n```\nuvicorn backend.api:app --reload\n```"
        )

    st.divider()

    # ── Model dropdown ────────────────────────────────────────────────────────
    st.subheader("Model Checkpoint")
    models_data = _fetch_models()
    model_names = [m["name"] for m in models_data] if models_data else ["(Không tìm thấy model)"]

    selected_model_name = st.selectbox(
        "Chọn checkpoint",
        options=model_names,
        key="model_select",
        help="Model sẽ được load lần đầu tiên và cache cho các lần sau.",
    )

    if models_data and selected_model_name in [m["name"] for m in models_data]:
        model_info = next(m for m in models_data if m["name"] == selected_model_name)
        st.caption(f"File: `{os.path.basename(model_info['model_path'])}`")

    st.divider()

    # ── Sample dropdown ───────────────────────────────────────────────────────
    st.subheader("Reference Sample")
    samples_data = _fetch_samples()
    sample_names = [s["name"] for s in samples_data] + ["Custom Upload"]

    selected_sample_name = st.selectbox(
        "Chọn sample tham chiếu",
        options=sample_names,
        key="sample_select",
        help="Chọn sample có sẵn hoặc upload file WAV của riêng bạn.",
    )

    st.divider()
    st.caption(f"API: `{API_BASE_URL}`")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main area
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🎙️ F5-TTS Vietnamese Demo")
st.caption("Tổng hợp giọng nói tiếng Việt với voice cloning")
st.divider()

col_left, col_right = st.columns(2, gap="large")

# ═══════════════════════════════════════════════════════════════════════════════
#  LEFT COLUMN — Reference audio & text
# ═══════════════════════════════════════════════════════════════════════════════
with col_left:
    st.subheader("🎤 Reference Audio")

    ref_audio_bytes: Optional[bytes] = None
    ref_text: str = ""

    if selected_sample_name == "Custom Upload":
        # ── Custom upload mode ────────────────────────────────────────────────
        uploaded_file = st.file_uploader(
            "Upload file WAV tham chiếu",
            type=["wav"],
            key="custom_wav",
            help="Chỉ chấp nhận định dạng .wav",
        )
        if uploaded_file is not None:
            ref_audio_bytes = uploaded_file.read()
            st.audio(ref_audio_bytes, format="audio/wav")
            st.caption(f"{uploaded_file.name} — {len(ref_audio_bytes):,} bytes")
        else:
            st.info("Chưa có file nào được upload.")

        st.divider()
        ref_text = st.text_area(
            "Transcript của reference audio",
            placeholder="Nhập nội dung lời thoại trong file wav...",
            height=240,
            key="custom_ref_text",
            help="Transcript phải khớp với nội dung trong file WAV.",
        )

    else:
        # ── Preset sample mode ────────────────────────────────────────────────
        sample_info = next(
            (s for s in samples_data if s["name"] == selected_sample_name), None
        )
        if sample_info:
            try:
                with open(sample_info["audio_path"], "rb") as f:
                    ref_audio_bytes = f.read()
                st.audio(ref_audio_bytes, format="audio/wav")
                st.caption(
                    f"{os.path.basename(sample_info['audio_path'])} — "
                    f"{len(ref_audio_bytes):,} bytes"
                )
            except OSError as e:
                st.error(f"Không đọc được file audio: {e}")

            st.divider()
            st.write("**Reference Text:**")
            ref_text = sample_info["text_content"]
            st.text_area(
                "Nội dung transcript",
                value=ref_text,
                height=240,
                key="preset_ref_text",
                disabled=True,
            )
        else:
            st.warning("Không tìm thấy sample. API có thể chưa chạy.")


# ═══════════════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN — Target text & Generate
# ═══════════════════════════════════════════════════════════════════════════════
with col_right:
    st.subheader("📝 Nội dung cần đọc")

    target_text = st.text_area(
        "Target Text",
        placeholder="Nhập đoạn văn bản muốn model đọc thành giọng nói...",
        height=200,
        key="target_text",
        help="Model sẽ đọc đoạn text này với giọng của reference audio.",
    )

    char_count = len(target_text) if target_text else 0
    st.caption(f"{char_count} ký tự")

    st.divider()

    # ── Validation hints ──────────────────────────────────────────────────────
    can_generate = (
        api_ok
        and ref_audio_bytes is not None
        and ref_text.strip() != ""
        and target_text.strip() != ""
        and models_data
        and selected_model_name in [m["name"] for m in models_data]
    )

    if not api_ok:
        st.warning("Backend API chưa khởi động. Không thể generate.")
    elif ref_audio_bytes is None:
        st.info("Vui lòng chọn hoặc upload reference audio.")
    elif not target_text.strip():
        st.info("Vui lòng nhập nội dung target text.")

    generate_btn = st.button(
        "🚀 Generate Speech",
        key="generate_btn",
        disabled=not can_generate,
        use_container_width=True,
        type="primary",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Generate logic
# ═══════════════════════════════════════════════════════════════════════════════

if generate_btn and can_generate:
    st.divider()
    st.subheader("⚡ Đang xử lý")

    # ── Encode audio ──────────────────────────────────────────────────────────
    ref_audio_b64 = _encode_audio_b64(ref_audio_bytes)  # type: ignore[arg-type]

    # ── Submit job ────────────────────────────────────────────────────────────
    try:
        with st.spinner("Đang gửi request lên API..."):
            submit_resp = _api_post(
                "/api/tts",
                {
                    "model_name": selected_model_name,
                    "ref_audio_b64": ref_audio_b64,
                    "ref_text": ref_text,
                    "target_text": target_text,
                },
            )
        job_id = submit_resp.get("job_id")
        if not job_id:
            st.error("API không trả về job_id.")
            st.stop()

        st.info(f"Job ID: `{job_id}`")
        logger.info("Submitted job %s", job_id)

    except requests.exceptions.ConnectionError:
        st.error(
            "Không kết nối được tới backend. "
            "Hãy đảm bảo API đang chạy trên cổng 8000."
        )
        st.stop()
    except requests.exceptions.HTTPError as exc:
        st.error(f"Lỗi HTTP từ API: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Lỗi không xác định: {exc}")
        st.stop()

    # ── Polling ───────────────────────────────────────────────────────────────
    final_result = _poll_job(job_id)

    # ── Hiển thị kết quả ─────────────────────────────────────────────────────
    if final_result.get("status") == "completed":
        st.success("Tổng hợp giọng nói thành công!")
        _display_audio_result(final_result)
    else:
        error_msg = final_result.get("error", "Unknown error")
        st.error(f"Tổng hợp thất bại:\n\n{error_msg}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "F5-TTS Vietnamese Demo — "
    f"[API Docs]({API_BASE_URL}/docs) | "
    f"[ReDoc]({API_BASE_URL}/redoc)"
)
