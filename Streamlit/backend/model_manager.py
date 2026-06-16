"""
Model Manager cho F5-TTS.

Quản lý vòng đời của các TTS model: scan checkpoints, load, cache,
unload khi đổi model. Tránh load lại model không cần thiết.
"""

import logging
import os
import sys
import threading
from typing import Optional

from .schemas import ModelInfo

logger = logging.getLogger(__name__)

# ─── Đường dẫn project gốc ───────────────────────────────────────────────────
# Streamlit/ -> TTS_DATA (2 levels up)
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(_THIS_DIR)  # /home/reg/TTS_DATA

MODELS_DIR = os.path.join(BASE_DIR, "models")
F5_TTS_SRC = os.path.join(BASE_DIR, "F5-TTS-Vietnamese", "src")

# Đảm bảo F5-TTS source và project root có trong sys.path
for _path in (F5_TTS_SRC, BASE_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def scan_models(models_dir: str = MODELS_DIR) -> list[ModelInfo]:
    """Quét thư mục models và trả về danh sách ModelInfo.

    Quy tắc nhận dạng checkpoint:
    - Thư mục f5-tts-v0  → model.pt
    - Thư mục f5-tts-NNNN → model_NNNN.pt
    Vocab được tìm theo thứ tự: vocab.txt trong thư mục checkpoint,
    rồi vocab.txt ngoài thư mục models.

    Args:
        models_dir: Đường dẫn tới thư mục chứa các checkpoint.

    Returns:
        Danh sách ModelInfo đã sắp xếp theo tên.
    """
    infos: list[ModelInfo] = []

    if not os.path.isdir(models_dir):
        logger.warning("Models directory không tồn tại: %s", models_dir)
        return infos

    for entry in sorted(os.listdir(models_dir)):
        full = os.path.join(models_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry == "samples":
            continue

        # Tìm file .pt
        model_file: Optional[str] = None
        if entry == "f5-tts-v0":
            candidate = os.path.join(full, "model.pt")
            if os.path.isfile(candidate):
                model_file = candidate
        else:
            # Lấy step number từ tên thư mục, vd: f5-tts-70000 → 70000
            parts = entry.split("-")
            step = parts[-1] if parts else ""
            candidate = os.path.join(full, f"model_{step}.pt")
            if os.path.isfile(candidate):
                model_file = candidate
            else:
                # Fallback: tìm bất kỳ .pt nào
                for fname in os.listdir(full):
                    if fname.endswith(".pt"):
                        model_file = os.path.join(full, fname)
                        break

        if model_file is None:
            logger.debug("Bỏ qua %s: không tìm thấy file .pt", entry)
            continue

        # Tìm vocab.txt
        vocab_file: Optional[str] = None
        local_vocab = os.path.join(full, "vocab.txt")
        fallback_vocab = os.path.join(models_dir, "vocab.txt")
        if os.path.isfile(local_vocab):
            vocab_file = local_vocab
        elif os.path.isfile(fallback_vocab):
            vocab_file = fallback_vocab
        else:
            logger.warning("Không tìm thấy vocab.txt cho model %s", entry)
            continue

        infos.append(
            ModelInfo(name=entry, model_path=model_file, vocab_path=vocab_file)
        )
        logger.debug("Đã tìm thấy model: %s → %s", entry, model_file)

    return infos


def scan_samples(samples_dir: Optional[str] = None) -> list[dict]:
    """Quét thư mục samples và trả về danh sách cặp (wav, txt).

    Args:
        samples_dir: Đường dẫn thư mục samples. Mặc định: models/samples.

    Returns:
        List[dict] với keys: name, audio_path, text_content, wps.
    """
    if samples_dir is None:
        samples_dir = os.path.join(MODELS_DIR, "samples")

    results = []
    if not os.path.isdir(samples_dir):
        logger.warning("Samples directory không tồn tại: %s", samples_dir)
        return results

    wav_files = {
        os.path.splitext(f)[0]: os.path.join(samples_dir, f)
        for f in os.listdir(samples_dir)
        if f.lower().endswith(".wav")
    }

    for stem, wav_path in sorted(wav_files.items()):
        txt_path = os.path.join(samples_dir, f"{stem}.txt")
        if not os.path.isfile(txt_path):
            logger.debug("Không có .txt tương ứng cho %s, bỏ qua.", stem)
            continue
        try:
            with open(txt_path, encoding="utf-8") as fh:
                text_content = fh.read().strip()
        except OSError as exc:
            logger.warning("Không đọc được %s: %s", txt_path, exc)
            continue

        # Tính toán WPS
        wps = 0.0
        try:
            import soundfile as sf
            info = sf.info(wav_path)
            duration = float(info.duration)
            words = len(text_content.strip().split())
            if duration > 0:
                wps = round(words / duration, 2)
        except Exception as exc:
            logger.debug("Soundfile failed for %s, trying torchaudio: %s", wav_path, exc)
            try:
                import torchaudio
                audio, sr = torchaudio.load(wav_path)
                duration = float(audio.shape[-1] / sr)
                words = len(text_content.strip().split())
                if duration > 0:
                    wps = round(words / duration, 2)
            except Exception as e:
                logger.warning("Không tính được WPS cho %s: %s", wav_path, e)

        results.append(
            {
                "name": stem,
                "audio_path": wav_path,
                "text_content": text_content,
                "wps": wps,
            }
        )

    return results


class ModelManager:
    """Singleton quản lý cache model F5-TTS.

    Chỉ giữ tối đa 1 model trong bộ nhớ. Khi đổi model, unload
    model cũ rồi load model mới. Thread-safe nhờ RLock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current_name: Optional[str] = None
        self._current_model = None  # instance của F5TTSVietnamese

    # ── Public API ────────────────────────────────────────────────────────────

    def get_model(self, model_info: ModelInfo):
        """Trả về model đang được cache. Load mới nếu tên khác.

        Args:
            model_info: ModelInfo chứa tên, đường dẫn checkpoint và vocab.

        Returns:
            Instance F5TTSVietnamese đã sẵn sàng cho inference.

        Raises:
            RuntimeError: Khi không load được model.
        """
        with self._lock:
            if self._current_name == model_info.name and self._current_model is not None:
                logger.info("Cache hit: đang dùng model '%s'", model_info.name)
                return self._current_model

            # Unload model cũ trước
            self._unload()

            logger.info(
                "Loading model '%s' từ %s ...", model_info.name, model_info.model_path
            )
            try:
                from components.tts.F5_TTS import F5TTSVietnamese  # type: ignore

                model = F5TTSVietnamese(
                    ckpt_file=model_info.model_path,
                    vocab_file=model_info.vocab_path,
                )
                self._current_model = model
                self._current_name = model_info.name
                logger.info("Load thành công model '%s'", model_info.name)
                return model
            except Exception as exc:
                logger.error("Lỗi load model '%s': %s", model_info.name, exc)
                raise RuntimeError(f"Không load được model '{model_info.name}': {exc}") from exc

    def current_name(self) -> Optional[str]:
        """Trả về tên model đang được cache, hoặc None nếu chưa có."""
        with self._lock:
            return self._current_name

    # ── Internal ──────────────────────────────────────────────────────────────

    def _unload(self) -> None:
        """Giải phóng model hiện tại khỏi bộ nhớ."""
        if self._current_model is not None:
            logger.info("Unloading model '%s'", self._current_name)
            try:
                import torch
                del self._current_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.warning("Lỗi khi unload model: %s", exc)
            finally:
                self._current_model = None
                self._current_name = None


# ── Module-level singleton ────────────────────────────────────────────────────
model_manager = ModelManager()
