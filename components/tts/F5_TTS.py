"""
F5-TTS Vietnamese model wrappers.

`F5TTSVietnamese` loads the original F5-TTS architecture.
`SemanticF5TTSVietnamese` loads the Custom_TTS semantic architecture.
Both share the same synthesis interface through `BaseF5TTSVietnamese`.
"""

import logging
import os
import tempfile
import wave
from typing import Callable, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .base import TTSModel, concat_audio, split_text

logger = logging.getLogger(__name__)


def calculate_duration_by_wps(
    gen_text: str,
    ref_text: str,
    ref_audio_duration: float,
    speed: float = 1.0,
    fallback_wps: float = 2.5,
) -> float:
    """Estimate total duration from the reference words-per-second ratio."""
    ref_words = len(ref_text.strip().split())
    gen_words = len(gen_text.strip().split())

    if ref_words > 0 and ref_audio_duration > 0:
        ref_wps = ref_words / ref_audio_duration
    else:
        ref_wps = fallback_wps

    local_speed = speed
    if gen_words < 3:
        local_speed = 0.3

    gen_duration = (gen_words / ref_wps) / local_speed
    gen_duration = gen_duration * (ref_wps / 4.0)
    return ref_audio_duration + gen_duration


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

_CKPT_PRIORITY = [
    "f5-tts-last",
    "f5-tts-72000",
    "f5-tts-71000",
    "f5-tts-70000",
    "f5-tts-60000",
    "f5-tts-50000",
    "f5-tts-v0",
]


def _find_default_ckpt() -> Optional[str]:
    for name in _CKPT_PRIORITY:
        folder = os.path.join(_MODELS_DIR, name)
        if not os.path.isdir(folder):
            continue
        pt_files = sorted(f for f in os.listdir(folder) if f.endswith(".pt"))
        if pt_files:
            return os.path.join(folder, pt_files[0])
    return None


def convert_to_pcm_wav(input_path: str) -> str:
    """Convert audio to a temporary PCM 16-bit WAV file."""
    audio, sr = torchaudio.load(input_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_wav_path = tmp.name

    try:
        audio_np = audio.cpu().numpy()
        if audio_np.ndim > 1 and audio_np.shape[0] > 1:
            audio_np = audio_np.T

        pcm_data = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
        n_channels = audio_np.shape[1] if audio_np.ndim > 1 else 1

        with wave.open(temp_wav_path, "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(pcm_data.tobytes())

        return temp_wav_path
    except Exception:
        if os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
            except OSError:
                pass
        raise


class BaseF5TTSVietnamese(TTSModel):
    """Shared inference flow for F5-style Vietnamese TTS wrappers."""

    def __init__(
        self,
        ckpt_file: Optional[str] = None,
        vocab_file: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        vocoder_name: str = "vocos",
        speed: float = 1.0,
        device: Optional[str] = None,
    ):
        if ckpt_file is None:
            ckpt_file = _find_default_ckpt()
            if ckpt_file is None:
                raise FileNotFoundError(
                    f"Khong tim thay checkpoint .pt nao trong {_MODELS_DIR}. "
                    "Vui long truyen ckpt_file cu the."
                )

        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file or self._resolve_vocab_file(ckpt_file)
        self.ref_audio_path = ref_audio
        self.ref_text = ref_text
        self.vocoder_name = vocoder_name
        self.speed = speed
        self.sample_rate = 24000

        if not os.path.exists(self.ckpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_file}")
        if not os.path.exists(self.vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_file}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _vocab_candidates(self, ckpt_file: str):
        return [
            os.path.join(os.path.dirname(ckpt_file), "vocab.txt"),
            os.path.join(_MODELS_DIR, "vocab.txt"),
        ]

    def _resolve_vocab_file(self, ckpt_file: str) -> str:
        for candidate in self._vocab_candidates(ckpt_file):
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Khong tim thay vocab.txt tu dong gan {ckpt_file}. "
            "Vui long truyen vocab_file cu the."
        )

    def _load_model(self):
        raise NotImplementedError

    def _infer_single(
        self,
        gen_text: str,
        ref_audio_processed,
        ref_text_processed: str,
        custom_duration_fn: Optional[Callable] = None,
        nfe_step: int = 32,
    ) -> Tuple[np.ndarray, int]:
        fix_duration = None
        if custom_duration_fn is not None:
            if custom_duration_fn == "wps":
                custom_duration_fn = calculate_duration_by_wps
            try:
                info = sf.info(ref_audio_processed)
                ref_audio_duration = float(info.duration)
                fix_duration = custom_duration_fn(
                    gen_text,
                    ref_text_processed,
                    ref_audio_duration,
                    self.speed,
                )
                logger.info("Custom duration: %.2fs for text '%s'", fix_duration, gen_text)
            except Exception as exc:
                logger.error("Loi khi tinh custom duration: %s", exc)

        audio_segment, final_sample_rate, _ = self._infer_process(
            ref_audio=ref_audio_processed,
            ref_text=ref_text_processed,
            gen_text=gen_text,
            model_obj=self.model,
            vocoder=self.vocoder,
            mel_spec_type=self.vocoder_name,
            speed=self.speed,
            device=self.device,
            progress=None,
            show_info=lambda x: None,
            fix_duration=fix_duration,
            nfe_step=nfe_step,
        )
        if isinstance(audio_segment, torch.Tensor):
            audio_segment = audio_segment.cpu().numpy()
        return audio_segment.astype(np.float32), final_sample_rate

    def synthesize(
        self,
        gen_text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        split_sentences: bool = False,
        min_words: int = 10,
        custom_duration_fn: Optional[Callable] = None,
        nfe_step: int = 64,
    ) -> Tuple[np.ndarray, int]:
        ref_audio_used = ref_audio_path or self.ref_audio_path
        ref_text_used = ref_text or self.ref_text

        if not ref_audio_used:
            raise ValueError("Khong co reference audio.")
        if not ref_text_used:
            raise ValueError("Khong co reference text.")

        pcm_ref_wav_path = None
        ref_audio_processed = None
        try:
            pcm_ref_wav_path = convert_to_pcm_wav(ref_audio_used)
            ref_text_lower = ref_text_used.lower()
            ref_audio_processed, ref_text_processed = self._preprocess_ref_audio_text(
                pcm_ref_wav_path,
                ref_text_lower,
                show_info=lambda x: None,
            )

            if split_sentences:
                chunks = split_text(gen_text, min_words=min_words)
                segments = []
                final_sample_rate = self.sample_rate
                for chunk in chunks:
                    seg, final_sample_rate = self._infer_single(
                        chunk.lower(),
                        ref_audio_processed,
                        ref_text_processed,
                        custom_duration_fn,
                        nfe_step=nfe_step,
                    )
                    segments.append(seg)
                return concat_audio(segments, final_sample_rate), final_sample_rate

            return self._infer_single(
                gen_text.lower(),
                ref_audio_processed,
                ref_text_processed,
                custom_duration_fn,
                nfe_step=nfe_step,
            )
        finally:
            for path in (pcm_ref_wav_path, ref_audio_processed):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass


class F5TTSVietnamese(BaseF5TTSVietnamese):
    """Original F5-TTS Vietnamese wrapper."""

    def _load_model(self):
        from omegaconf import OmegaConf

        from f5_tts.infer.utils_infer import (
            infer_process,
            load_model,
            load_vocoder,
            preprocess_ref_audio_text,
        )
        from f5_tts.model import DiT

        self.vocoder = load_vocoder(
            vocoder_name=self.vocoder_name,
            is_local=False,
            local_path="",
            device=self.device,
        )

        import f5_tts

        if hasattr(f5_tts, "__path__"):
            f5_tts_path = list(f5_tts.__path__)[0]
        else:
            f5_tts_path = os.path.dirname(f5_tts.__file__) if f5_tts.__file__ else ""

        model_cfg_path = os.path.join(f5_tts_path, "configs", "F5TTS_Base.yaml")
        model_cfg = OmegaConf.load(model_cfg_path).model

        self.model = load_model(
            model_cls=DiT,
            model_cfg=model_cfg.arch,
            ckpt_path=self.ckpt_file,
            mel_spec_type=self.vocoder_name,
            vocab_file=self.vocab_file,
            device=self.device,
        )
        self._infer_process = infer_process
        self._preprocess_ref_audio_text = preprocess_ref_audio_text


class SemanticF5TTSVietnamese(BaseF5TTSVietnamese):
    """Custom_TTS SemanticF5TTS wrapper."""

    def _vocab_candidates(self, ckpt_file: str):
        return [
            os.path.join(os.path.dirname(ckpt_file), "vocab.txt"),
            "/home/reg/TTS_DATA/Custom_TTS/data/my_dataset/vocab.txt",
            "/home/reg/TTS_DATA/models/f5-tts-v0/vocab.txt",
            os.path.join(_MODELS_DIR, "vocab.txt"),
        ]

    def _load_model(self):
        from importlib.resources import files

        from omegaconf import OmegaConf

        from custom_tts.infer.utils_infer import (
            infer_process,
            load_semantic_model,
            load_vocoder,
            preprocess_ref_audio_text,
        )
        from custom_tts.model import DiT

        model_cfg_path = files("custom_tts").joinpath("configs/SemanticF5TTS_Base.yaml")
        model_cfg = OmegaConf.load(str(model_cfg_path)).model
        self.vocoder_name = model_cfg.mel_spec.mel_spec_type
        self.sample_rate = model_cfg.mel_spec.target_sample_rate

        self.vocoder = load_vocoder(
            vocoder_name=self.vocoder_name,
            is_local=False,
            local_path="",
            device=self.device,
        )
        self.model = load_semantic_model(
            model_cls=DiT,
            model_cfg=model_cfg.arch,
            semantic_cfg=model_cfg.semantic,
            ckpt_path=self.ckpt_file,
            mel_spec_type=self.vocoder_name,
            vocab_file=self.vocab_file,
            device=self.device,
        )
        self._infer_process = infer_process
        self._preprocess_ref_audio_text = preprocess_ref_audio_text
        logger.info("Loaded SemanticF5TTS checkpoint: %s", self.ckpt_file)
