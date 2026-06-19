"""
F5-TTS & OmniVoice Demo — Streamlit Frontend.

Giao diện cho phép:
  1. Sử dụng model F5-TTS hoặc OmniVoice.
  2. Hỗ trợ chế độ Voice Clone và Voice Design cho OmniVoice.
  3. Chọn sample tham chiếu hoặc upload sample tùy chỉnh.
  4. Nhập text cần tổng hợp giọng nói.
  5. Hiển thị kết quả âm thanh tổng hợp.

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
st_logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL_SEC = 2       # Polling mỗi 2 giây
MAX_POLL_ATTEMPTS = 150     # Tối đa ~5 phút

# ─── OmniVoice Constants ──────────────────────────────────────────────────────
OMNIVOICE_LANGUAGES = [
    "Auto", "Vietnamese", "English", "Chinese", "Japanese", "Korean",
    "French", "German", "Spanish", "Russian", "Portuguese"
]

OMNIVOICE_GENDER = ["Auto", "Male / 男", "Female / 女"]

OMNIVOICE_AGE = [
    "Auto",
    "Child / 儿童",
    "Teenager / 少年",
    "Young Adult / 青年",
    "Middle-aged / 中年",
    "Elderly / 老年",
]

OMNIVOICE_PITCH = [
    "Auto",
    "Very Low Pitch / 极低音调",
    "Low Pitch / 低音调",
    "Moderate Pitch / 中音调",
    "High Pitch / 高音调",
    "Very High Pitch / 极高音调",
]

OMNIVOICE_STYLE = ["Auto", "Whisper / 耳语"]

OMNIVOICE_ACCENTS = [
    "Auto",
    "American Accent / 美式口音",
    "Australian Accent / 澳大利亚口音",
    "British Accent / 英国口音",
    "Chinese Accent / 中国口音",
    "Canadian Accent / 加拿大口音",
    "Indian Accent / 印度口音",
    "Korean Accent / 韩国口音",
    "Portuguese Accent / 葡萄牙口音",
    "Russian Accent / 俄罗斯口音",
    "Japanese Accent / 日本口音",
]

OMNIVOICE_DIALECTS = [
    "Auto",
    "Henan Dialect / 河南话",
    "Shaanxi Dialect / 陕西话",
    "Sichuan Dialect / 四川话",
    "Guizhou Dialect / 贵州话",
    "Yunnan Dialect / 云南话",
    "Guilin Dialect / 桂林话",
    "Jinan Dialect / 济南话",
    "Shijiazhuang Dialect / 石家庄话",
    "Gansu Dialect / 甘肃话",
    "Ningxia Dialect / 宁夏话",
    "Qingdao Dialect / 青岛话",
    "Northeast Dialect / 东北话",
]

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual TTS Demo (F5-TTS & OmniVoice)",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Nhãn model */
.model-label {
    background: linear-gradient(135deg, #f093fb, #f5576c);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-block;
    margin-bottom: 8px;
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
    """Lấy danh sách models từ API."""
    try:
        data = _api_get("/api/models")
        return data.get("models", [])
    except Exception as exc:
        st_logger.warning("Không lấy được danh sách models: %s", exc)
        return []


def _fetch_samples() -> list[dict]:
    """Lấy danh sách samples từ API. Trả về [] nếu lỗi."""
    try:
        data = _api_get("/api/samples")
        return data.get("samples", [])
    except Exception as exc:
        st_logger.warning("Không lấy được danh sách samples: %s", exc)
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


def _poll_single_job(
    job_id: str,
) -> dict:
    """Polling job cho đến khi hoàn thành hoặc lỗi.

    Returns:
        result — dict kết quả cuối cùng của job.
    """
    progress_bar = st.progress(0, text="Đang khởi tạo...")

    TERMINAL = {"completed", "failed"}
    result: dict = {}

    for attempt in range(MAX_POLL_ATTEMPTS):
        r = _job_status(job_id)
        done = r.get("status") in TERMINAL

        progress = 0.0
        if done:
            progress = 1.0
        else:
            progress = min((attempt + 1) / MAX_POLL_ATTEMPTS * 0.9, 0.9)

        progress_bar.progress(progress, text="Đang xử lý...")

        if done:
            result = r
            progress_bar.progress(1.0, text="Hoàn thành!")
            break

        time.sleep(POLL_INTERVAL_SEC)
    else:
        if not result:
            result = {"status": "failed", "error": "Polling timeout"}

    return result


def _display_single_result(
    result: dict,
    model_name: str,
) -> None:
    """Hiển thị kết quả model."""
    st.subheader("🎵 Kết quả")

    st.markdown(
        f'<span class="model-label">🎙️ {model_name}</span>',
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar — Model & Sample selector
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Cấu hình")
    st.caption("Chọn model và các tùy chọn tương ứng")
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

    # ── Model Checkpoint ──────────────────────────────────────────────────────
    st.subheader("Model Checkpoint")
    models_data = _fetch_models()
    model_choices = ["f5-tts-70000", "omnivoice"]
    selected_model_name = st.selectbox(
        "Chọn Model Checkpoint",
        options=model_choices,
        index=0,
        key="selected_model_name"
    )

    if selected_model_name == "omnivoice":
        st.info("Model: **OmniVoice**")
        st.caption("Hỗ trợ 600+ ngôn ngữ, Voice Clone & Voice Design.")
        omnivoice_mode = st.selectbox(
            "Chế độ OmniVoice",
            options=["Voice Clone", "Voice Design"],
            index=0,
            key="omnivoice_mode"
        )
    else:
        st.info(f"Model: **{selected_model_name}**")
        if models_data:
            model_info = next((m for m in models_data if m["name"] == selected_model_name), None)
            if model_info:
                st.caption(f"File: `{os.path.basename(model_info['model_path'])}`")

    st.divider()

    # ── Sample dropdown (Only for F5-TTS or OmniVoice Voice Clone) ──────────────
    show_sample_selector = (
        selected_model_name == "f5-tts-70000" or
        (selected_model_name == "omnivoice" and omnivoice_mode == "Voice Clone")
    )

    if show_sample_selector:
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

    # ── Số bước chạy DiT & Tùy chọn nâng cao (F5-TTS chỉ định) ─────────────────
    if selected_model_name == "f5-tts-70000":
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
#  LEFT COLUMN — Reference audio & text / Voice Design attributes
# ═══════════════════════════════════════════════════════════════════════════════
with col_left:
    ref_audio_bytes: Optional[bytes] = None
    ref_text: str = ""

    # -- 1. Reference Audio (F5-TTS or OmniVoice Voice Clone) --
    if selected_model_name == "f5-tts-70000" or (selected_model_name == "omnivoice" and omnivoice_mode == "Voice Clone"):
        st.subheader("🎤 Reference Audio")

        if selected_sample_name == "Custom Upload":
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
                "Transcript của reference audio (để trống để OmniVoice tự nhận dạng)",
                placeholder="Nhập nội dung lời thoại trong file wav...",
                height=200,
                key="custom_ref_text",
                help="Transcript phải khớp với nội dung trong file WAV. OmniVoice có thể tự nhận dạng nếu để trống.",
            )

            # Tính toán WPS động cho Custom Upload
            if ref_audio_bytes is not None and ref_text.strip():
                try:
                    import soundfile as _sf
                    import io
                    _info = _sf.info(io.BytesIO(ref_audio_bytes))
                    _duration = float(_info.duration)
                    _words = len(ref_text.strip().split())
                    if _duration > 0:
                        custom_wps = _words / _duration
                        st.info(f"📊 **Custom Audio WPS:** `{custom_wps:.2f}` từ/giây (Số từ: {_words}, Thời lượng: {_duration:.2f}s)")
                except Exception:
                    pass
        else:
            # Preset sample
            sample_info = next(
                (s for s in samples_data if s["name"] == selected_sample_name), None
            )
            if sample_info:
                try:
                    with open(sample_info["audio_path"], "rb") as f:
                        ref_audio_bytes = f.read()
                    st.audio(ref_audio_bytes, format="audio/wav")
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
                st.text_area(
                    "Nội dung transcript",
                    value=ref_text,
                    height=200,
                    key=f"preset_ref_text_{selected_sample_name}",
                    disabled=True,
                )
            else:
                st.warning("Không tìm thấy sample. API có thể chưa chạy.")

        # OmniVoice Clone Settings (chỉ hiện khi đang ở chế độ Clone)
        if selected_model_name == "omnivoice":
            st.divider()
            st.subheader("⚙️ OmniVoice Clone Settings")
            ov_lang = st.selectbox("Language / Ngôn ngữ", OMNIVOICE_LANGUAGES, key="ov_clone_lang")
            ov_instruct = st.text_input(
                "Instruct (optional) / Lời nhắc tùy chọn",
                placeholder="VD: Speak with excitement",
                key="ov_clone_instruct",
            )
            with st.expander("⚙️ Advanced Settings", expanded=False):
                ov_ns = st.slider("Inference Steps", 4, 64, 32, key="ov_clone_ns")
                ov_gs = st.slider("Guidance Scale (CFG)", 0.0, 4.0, 2.0, 0.1, key="ov_clone_gs")
                ov_dn = st.checkbox("Denoise", value=True, key="ov_clone_dn")
                ov_sp = st.slider("Speed", 0.5, 1.5, 1.0, 0.05, key="ov_clone_sp")
                ov_du = st.number_input("Duration (seconds)", value=0.0, step=0.5, key="ov_clone_du", help="0.0 = tự động.")
                ov_pp = st.checkbox("Preprocess Prompt", value=True, key="ov_clone_pp")
                ov_po = st.checkbox("Postprocess Output", value=True, key="ov_clone_po")

    # -- 2. OmniVoice Voice Design Mode --
    if selected_model_name == "omnivoice" and omnivoice_mode == "Voice Design":
        st.subheader("🎨 Voice Design Settings")

        ov_lang = st.selectbox("Language / Ngôn ngữ", OMNIVOICE_LANGUAGES, key="ov_design_lang")

        col_design_1, col_design_2 = st.columns(2)
        with col_design_1:
            ov_gender = st.selectbox("Gender / Giới tính", OMNIVOICE_GENDER, key="ov_design_gender")
            ov_age = st.selectbox("Age / Độ tuổi", OMNIVOICE_AGE, key="ov_design_age")
            ov_pitch = st.selectbox("Pitch / Tông giọng", OMNIVOICE_PITCH, key="ov_design_pitch")
        with col_design_2:
            ov_style = st.selectbox("Style / Phong cách", OMNIVOICE_STYLE, key="ov_design_style")
            ov_accent = st.selectbox("English Accent / Giọng tiếng Anh", OMNIVOICE_ACCENTS, key="ov_design_accent")
            ov_dialect = st.selectbox("Chinese Dialect / Giọng phương ngôn", OMNIVOICE_DIALECTS, key="ov_design_dialect")

        st.divider()
        with st.expander("⚙️ Advanced Settings", expanded=False):
            ov_ns = st.slider("Inference Steps", 4, 64, 32, key="ov_design_ns")
            ov_gs = st.slider("Guidance Scale (CFG)", 0.0, 4.0, 2.0, 0.1, key="ov_design_gs")
            ov_dn = st.checkbox("Denoise", value=True, key="ov_design_dn")
            ov_sp = st.slider("Speed", 0.5, 1.5, 1.0, 0.05, key="ov_design_sp")
            ov_du = st.number_input("Duration (seconds)", value=0.0, step=0.5, key="ov_design_du", help="0.0 = tự động.")
            ov_pp = st.checkbox("Preprocess Prompt", value=True, key="ov_design_pp")
            ov_po = st.checkbox("Postprocess Output", value=True, key="ov_design_po")


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
        help="Model sẽ đọc đoạn text này.",
    )

    word_count = len(target_text.strip().split()) if target_text else 0
    st.caption(f"{word_count} từ")

    # ── Validation hints ──────────────────────────────────────────────────────
    if selected_model_name == "f5-tts-70000":
        can_generate = (
            api_ok
            and ref_audio_bytes is not None
            and ref_text.strip() != ""
            and target_text.strip() != ""
        )
    elif selected_model_name == "omnivoice":
        if omnivoice_mode == "Voice Clone":
            can_generate = (
                api_ok
                and ref_audio_bytes is not None
                and target_text.strip() != ""
            )
        else:  # Voice Design
            can_generate = (
                api_ok
                and target_text.strip() != ""
            )
    else:
        can_generate = False

    if not api_ok:
        st.warning("Backend API chưa khởi động. Không thể tổng hợp giọng nói.")
    elif (selected_model_name == "f5-tts-70000" or (selected_model_name == "omnivoice" and omnivoice_mode == "Voice Clone")) and ref_audio_bytes is None:
        st.info("Vui lòng chọn hoặc upload reference audio.")
    elif not target_text.strip():
        st.info("Vui lòng nhập nội dung target text.")
    else:
        info_str = f"▶️ Sẽ chạy model: **{selected_model_name}**"
        if selected_model_name == "omnivoice":
            info_str += f" ({omnivoice_mode})"
        st.info(info_str)

    generate_btn = st.button(
        "🚀 Tổng hợp Giọng nói",
        key="generate_btn",
        disabled=not can_generate,
        use_container_width=True,
        type="primary",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Submit & Polling Logic
# ═══════════════════════════════════════════════════════════════════════════════

if generate_btn and can_generate:
    st.subheader("⚡ Đang xử lý")

    # ── Prepare payload ───────────────────────────────────────────────────────
    payload = {
        "model_name": selected_model_name,
        "target_text": target_text,
    }

    if selected_model_name == "f5-tts-70000":
        ref_audio_b64 = _encode_audio_b64(ref_audio_bytes)  # type: ignore[arg-type]
        payload.update({
            "ref_audio_b64": ref_audio_b64,
            "ref_text": ref_text,
            "split_sentences": bool(split_sentences),
            "min_words": int(min_words),
            "nfe_step": int(nfe_step),
        })
    elif selected_model_name == "omnivoice":
        if omnivoice_mode == "Voice Clone":
            ref_audio_b64 = _encode_audio_b64(ref_audio_bytes)  # type: ignore[arg-type]
            payload.update({
                "omnivoice_mode": "clone",
                "ref_audio_b64": ref_audio_b64,
                "ref_text": ref_text,
                "language": ov_lang,
                "instruct": ov_instruct,
                "nfe_step": int(ov_ns),
                "guidance_scale": float(ov_gs),
                "denoise": bool(ov_dn),
                "speed": float(ov_sp),
                "duration": float(ov_du) if ov_du > 0 else None,
                "preprocess_prompt": bool(ov_pp),
                "postprocess_output": bool(ov_po),
            })
        else:  # Voice Design
            payload.update({
                "omnivoice_mode": "design",
                "language": ov_lang,
                "gender": ov_gender,
                "age": ov_age,
                "pitch": ov_pitch,
                "style": ov_style,
                "english_accent": ov_accent,
                "chinese_dialect": ov_dialect,
                "nfe_step": int(ov_ns),
                "guidance_scale": float(ov_gs),
                "denoise": bool(ov_dn),
                "speed": float(ov_sp),
                "duration": float(ov_du) if ov_du > 0 else None,
                "preprocess_prompt": bool(ov_pp),
                "postprocess_output": bool(ov_po),
            })

    # ── Submit job qua endpoint /api/tts ──────────────────────────────────────
    try:
        with st.spinner("Đang gửi request lên API..."):
            submit_resp = _api_post(
                "/api/tts",
                payload,
            )

        job_id = submit_resp.get("job_id")

        if not job_id:
            st.error("API không trả về job_id.")
            st.stop()

        st_logger.info("TTS job submitted: %s (model=%s)", job_id, selected_model_name)

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

    # ── Polling job ────────────────────────────────────────────────────────
    result = _poll_single_job(job_id)

    # ── Hiển thị kết quả ───────────────────────────────────────────────────
    _display_single_result(result, selected_model_name)


# ─── Footer ───────────────────────────────────────────────────────────────────

st.caption(
    "F5-TTS & OmniVoice Vietnamese Demo | "
    f"[API Docs]({API_BASE_URL}/docs) | "
    f"[ReDoc]({API_BASE_URL}/redoc)"
)
