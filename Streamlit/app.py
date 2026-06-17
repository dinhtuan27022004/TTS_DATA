"""
F5-TTS Demo — Streamlit Frontend (Compare Mode).

Giao diện cho phép:
  1. Chọn checkpoint model F5-TTS (không bao gồm v0 — luôn là baseline).
  2. Chọn sample tham chiếu hoặc upload sample tùy chỉnh.
  3. Nhập text cần synthesize.
  4. Gửi request compare → chạy v0 + model đang chọn song song → hiển thị 2 kết quả.

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
BASELINE_MODEL = "f5-tts-v0"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F5-TTS Vietnamese Demo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Màu sắc nhận dạng 2 model */
.model-v0-label {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-block;
    margin-bottom: 8px;
}
.model-selected-label {
    background: linear-gradient(135deg, #f093fb, #f5576c);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-block;
    margin-bottom: 8px;
}
.compare-card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


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
    """Lấy danh sách models từ API (không bao gồm v0). Trả về [] nếu lỗi."""
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


def _job_status(job_id: str) -> dict:
    """Lấy trạng thái của một job."""
    try:
        return _api_get(f"/api/tts/{job_id}")
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _poll_compare_jobs(
    job_id_v0: str,
    job_id_selected: str,
    model_v0: str,
    model_selected: str,
) -> tuple[dict, dict]:
    """Polling cả 2 job cho đến khi cả 2 hoàn thành hoặc lỗi.

    Returns:
        (result_v0, result_selected) — dict kết quả cuối cùng của từng job.
    """
    progress_bar = st.progress(0, text="Đang khởi tạo so sánh...")

    TERMINAL = {"completed", "failed"}
    result_v0: dict = {}
    result_sel: dict = {}

    for attempt in range(MAX_POLL_ATTEMPTS):
        r_v0 = _job_status(job_id_v0)
        r_sel = _job_status(job_id_selected)

        done_v0 = r_v0.get("status") in TERMINAL
        done_sel = r_sel.get("status") in TERMINAL

        progress = 0.0
        if done_v0 and done_sel:
            progress = 1.0
        elif done_v0 or done_sel:
            progress = 0.6
        else:
            progress = min((attempt + 1) / MAX_POLL_ATTEMPTS * 0.5, 0.5)

        progress_bar.progress(progress, text="Đang xử lý so sánh...")

        if done_v0:
            result_v0 = r_v0
        if done_sel:
            result_sel = r_sel

        if done_v0 and done_sel:
            progress_bar.progress(1.0, text="Hoàn thành cả 2 model!")
            break

        time.sleep(POLL_INTERVAL_SEC)
    else:
        if not result_v0:
            result_v0 = {"status": "failed", "error": "Polling timeout"}
        if not result_sel:
            result_sel = {"status": "failed", "error": "Polling timeout"}

    return result_v0, result_sel


def _display_compare_results(
    result_v0: dict,
    result_sel: dict,
    model_v0: str,
    model_selected: str,
) -> None:
    """Hiển thị kết quả 2 model cạnh nhau để so sánh."""
    st.subheader("🎵 Kết quả so sánh")

    col_v0, col_sel = st.columns(2, gap="large")

    def _render_result(col, result: dict, model_name: str, label_class: str, emoji: str):
        with col:
            st.markdown(
                f'<span class="{label_class}">{emoji} {model_name}</span>',
                unsafe_allow_html=True,
            )
            if result.get("status") == "completed":
                audio_url = result.get("audio_url")
                duration = result.get("duration")
                if duration:
                    st.metric("Thời lượng", f"{duration:.2f}s")
                if audio_url:
                    full_url = f"{API_BASE_URL}{audio_url}"
                    try:
                        audio_resp = requests.get(full_url, timeout=30)
                        audio_resp.raise_for_status()
                        audio_bytes = audio_resp.content
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button(
                            label="⬇️ Tải WAV",
                            data=audio_bytes,
                            file_name=f"{model_name}_{os.path.basename(audio_url)}",
                            mime="audio/wav",
                            key=f"dl_{model_name}",
                        )
                    except Exception as exc:
                        st.error(f"Không tải được audio: {exc}")
                else:
                    st.warning("Không có đường dẫn audio.")
            elif result.get("status") == "failed":
                st.error(f"Thất bại: {result.get('error', 'Unknown error')}")
            else:
                st.warning(f"Trạng thái không xác định: {result.get('status')}")

    _render_result(col_v0, result_v0, model_v0, "model-v0-label", "🔵")
    _render_result(col_sel, result_sel, model_selected, "model-selected-label", "🔴")


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

    # ── Model dropdown (không có v0) ───────────────────────────────────────────
    st.subheader("Model Checkpoint")
    st.caption(
        f"Model **{BASELINE_MODEL}** luôn được chạy ngầm làm baseline để so sánh."
    )
    models_data = _fetch_models()
    def _model_step(name: str) -> int:
        """Trích step number từ tên model, vd: f5-tts-70000 → 70000."""
        try:
            return int(name.split("-")[-1])
        except ValueError:
            return 0

    model_names = (
        sorted([m["name"] for m in models_data], key=_model_step, reverse=True)
        if models_data else []
    )
    # Chỉ cho phép chọn f5-tts-70000 (ẩn/comment các model khác)
    model_names = [name for name in model_names if name == "f5-tts-70000"]

    if model_names:
        selected_model_name = st.selectbox(
            "Chọn checkpoint để so sánh với v0",
            options=model_names,
            key="model_select",
            help="Model này sẽ chạy cùng lúc với f5-tts-v0 để so sánh kết quả.",
        )
        if selected_model_name in [m["name"] for m in models_data]:
            model_info = next(m for m in models_data if m["name"] == selected_model_name)
            st.caption(f"File: `{os.path.basename(model_info['model_path'])}`")
    else:
        selected_model_name = None
        st.warning("Không tìm thấy model nào (ngoài v0). Hãy kiểm tra thư mục models/.")

    st.divider()

    # ── Sample dropdown ───────────────────────────────────────────────────────
    st.subheader("Reference Sample")
    samples_data = _fetch_samples()
    sample_names = [s["name"] for s in samples_data] + ["Custom Upload"]

    # Tạo mapping để hiển thị WPS bên cạnh tên mẫu
    sample_wps_map = {s["name"]: s.get("wps", 0.0) for s in samples_data}

    def format_sample_option(opt):
        if opt == "Custom Upload":
            return opt
        wps_val = sample_wps_map.get(opt, 0.0)
        return f"{opt} (WPS: {wps_val:.2f})"

    selected_sample_name = st.selectbox(
        "Chọn sample tham chiếu",
        options=sample_names,
        format_func=format_sample_option,
        key="sample_select",
        help="Chọn sample có sẵn hoặc upload file WAV của riêng bạn.",
    )

    st.divider()

    # ── Số bước chạy DiT ──────────────────────────────────────────────
    st.subheader("Số bước chạy DiT")
    nfe_step = st.slider(
        "Số bước (NFE Steps)",
        min_value=16,
        max_value=128,
        value=64,
        step=8,
        key="nfe_step",
        help="Tăng số bước chạy giúp âm thanh chi tiết hơn nhưng làm tăng thời gian suy luận (mặc định là 64).",
    )

    st.divider()

    # ── Tùy chọn nâng cao ───────────────────────────────────────────
    with st.expander("⚙️ Tùy chọn nâng cao", expanded=False):
        split_sentences = st.toggle(
            "Tách câu (Split Sentences)",
            value=False,
            key="split_sentences",
            help=(
                "Khi bật, target text sẽ được tách thành nhiều câu ngắn "
                "(dựa theo dấu câu), mỗi câu tổng hợp riêng rồi ghép lại thành "
                "một file WAV. Phù hợp với đoạn văn bản dài."
            ),
        )
        min_words = st.number_input(
            "Số từ tối thiểu / câu",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="min_words",
            disabled=not split_sentences,
            help="Câu ngắn hơn ngưỡng này sẽ được gộp với câu kế tiếp.",
        )

    st.divider()
    st.caption(f"API: `{API_BASE_URL}`")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main area
# ═══════════════════════════════════════════════════════════════════════════════

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
            uploaded_file.seek(0)
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

        # Tính toán WPS động cho Custom Upload
        if ref_audio_bytes is not None and ref_text.strip():
            try:
                import soundfile as sf
                import io
                info = sf.info(io.BytesIO(ref_audio_bytes))
                duration = float(info.duration)
                words = len(ref_text.strip().split())
                if duration > 0:
                    custom_wps = words / duration
                    st.info(f"📊 **Custom Audio WPS:** `{custom_wps:.2f}` từ/giây (Số từ: {words}, Thời lượng: {duration:.2f}s)")
            except Exception as exc:
                try:
                    import torchaudio
                    import io
                    audio, sr = torchaudio.load(io.BytesIO(ref_audio_bytes))
                    duration = float(audio.shape[-1] / sr)
                    words = len(ref_text.strip().split())
                    if duration > 0:
                        custom_wps = words / duration
                        st.info(f"📊 **Custom Audio WPS:** `{custom_wps:.2f}` từ/giây (Số từ: {words}, Thời lượng: {duration:.2f}s)")
                except Exception:
                    pass

    else:
        # ── Preset sample mode ────────────────────────────────────────────────
        sample_info = next(
            (s for s in samples_data if s["name"] == selected_sample_name), None
        )
        if sample_info:
            try:
                with open(sample_info["audio_path"], "rb") as f:
                    ref_audio_bytes = f.read()
                # Key động → Streamlit tạo lại audio player khi sample thay đổi
                st.audio(
                    ref_audio_bytes,
                    format="audio/wav",
                )
                wps_val = sample_info.get("wps", 0.0)
                st.caption(
                    f"{os.path.basename(sample_info['audio_path'])} — "
                    f"{len(ref_audio_bytes):,} bytes"
                )
                st.markdown(f"📊 **Reference WPS:** `{wps_val:.2f}` từ/giây")
            except OSError as e:
                st.error(f"Không đọc được file audio: {e}")

            st.divider()
            st.write("**Reference Text:**")
            ref_text = sample_info["text_content"]
            # Key động → Streamlit tạo lại text_area với value mới khi sample thay đổi
            # Nếu dùng key cố định, Streamlit giữ nguyên giá trị cũ trong session_state
            st.text_area(
                "Nội dung transcript",
                value=ref_text,
                height=240,
                key=f"preset_ref_text_{selected_sample_name}",
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
        help="Cả 2 model (v0 và model đang chọn) sẽ đọc đoạn text này.",
    )

    word_count = len(target_text.strip().split()) if target_text else 0
    st.caption(f"{word_count} từ")



    # ── Validation hints ──────────────────────────────────────────────────────
    can_generate = (
        api_ok
        and ref_audio_bytes is not None
        and ref_text.strip() != ""
        and target_text.strip() != ""
        and selected_model_name is not None
    )

    if not api_ok:
        st.warning("Backend API chưa khởi động. Không thể generate.")
    elif ref_audio_bytes is None:
        st.info("Vui lòng chọn hoặc upload reference audio.")
    elif not target_text.strip():
        st.info("Vui lòng nhập nội dung target text.")
    elif selected_model_name is None:
        st.warning("Không tìm thấy model nào để so sánh.")
    else:
        # Hiển thị thông tin về 2 model sẽ chạy
        st.info(
            f"▶️ Sẽ chạy **2 model** song song:\n\n"
            f"🔵 **{BASELINE_MODEL}** (baseline)\n\n"
            f"🔴 **{selected_model_name}** (model đang chọn)"
        )

    generate_btn = st.button(
        "🚀 So sánh 2 Model",
        key="generate_btn",
        disabled=not can_generate,
        use_container_width=True,
        type="primary",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Compare logic
# ═══════════════════════════════════════════════════════════════════════════════

if generate_btn and can_generate:

    st.subheader("⚡ Đang xử lý so sánh")

    # ── Encode audio ──────────────────────────────────────────────────────────
    ref_audio_b64 = _encode_audio_b64(ref_audio_bytes)  # type: ignore[arg-type]

    # ── Submit 2 jobs qua endpoint compare ───────────────────────────────────
    try:
        with st.spinner("Đang gửi request lên API..."):
            submit_resp = _api_post(
                "/api/tts/compare",
                {
                    "model_name": selected_model_name,
                    "ref_audio_b64": ref_audio_b64,
                    "ref_text": ref_text,
                    "target_text": target_text,
                    "split_sentences": bool(split_sentences),
                    "min_words": int(min_words),
                    "nfe_step": int(nfe_step),
                },
            )

        job_id_v0 = submit_resp.get("job_id_v0")
        job_id_selected = submit_resp.get("job_id_selected")
        model_v0 = submit_resp.get("model_v0", BASELINE_MODEL)
        model_selected = submit_resp.get("model_selected", selected_model_name)

        if not job_id_v0 or not job_id_selected:
            st.error("API không trả về job_id đầy đủ.")
            st.stop()

        logger.info("Compare jobs: v0=%s, sel=%s", job_id_v0, job_id_selected)

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

    # ── Polling 2 jobs ─────────────────────────────────────────────────────
    result_v0, result_sel = _poll_compare_jobs(
        job_id_v0, job_id_selected, model_v0, model_selected
    )

    # ── Hiển thị kết quả so sánh ──────────────────────────────────────────
    _display_compare_results(result_v0, result_sel, model_v0, model_selected)


# ─── Footer ───────────────────────────────────────────────────────────────────

st.caption(
    "F5-TTS Vietnamese Demo — Compare Mode | "
    f"[API Docs]({API_BASE_URL}/docs) | "
    f"[ReDoc]({API_BASE_URL}/redoc)"
)
