"""
F5-TTS Vietnamese Model Wrapper.

Implement TTSModel interface cho F5-TTS Vietnamese.
Load checkpoint và vocab từ local folder: models/f5-tts-v0/
"""

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

# Project paths (relative)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# F5-TTS source path
F5_TTS_DIR = os.path.join(BASE_DIR, "F5-TTS-Vietnamese")
F5_TTS_SRC = os.path.join(F5_TTS_DIR, "src")
F5_DEFAULT_REF_AUDIO = os.path.join(F5_TTS_DIR, "ref.wav")

# Local model files
MODEL_DIR = os.path.join(BASE_DIR, "models", "f5-tts-v0")
LOCAL_CKPT_FILE = os.path.join(MODEL_DIR, "model.pt")
LOCAL_VOCAB_FILE = os.path.join(MODEL_DIR, "vocab.txt")

# Thêm F5-TTS source vào path
if F5_TTS_SRC not in sys.path:
    sys.path.insert(0, F5_TTS_SRC)

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT

# Thêm project root vào path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluate.models import TTSModel


class F5TTSVietnamese(TTSModel):
    """F5-TTS Vietnamese model wrapper.

    Load checkpoint và vocab từ models/f5-tts-v0/.
    Output sample rate: 24000 Hz.

    Args:
        ckpt_file: Đường dẫn đến file checkpoint (.pt). None = models/f5-tts-v0/model.pt
        vocab_file: Đường dẫn đến file vocab (.txt). None = models/f5-tts-v0/vocab.txt
        ref_audio: Đường dẫn đến file audio tham chiếu mặc định (.wav).
        ref_text: Transcript của audio tham chiếu mặc định.
        vocoder_name: Tên vocoder ("vocos" hoặc "bigvgan").
        speed: Tốc độ tổng hợp (1.0 = bình thường).
        device: Device để chạy model ("cuda", "cpu", hoặc None = auto).
    """

    def __init__(
        self,
        ckpt_file: Optional[str] = None,
        vocab_file: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: str = "cả hai bên hãy cố gắng hiểu cho nhau",
        vocoder_name: str = "vocos",
        speed: float = 1.0,
        device: Optional[str] = None,
    ):
        self.ckpt_file = ckpt_file or LOCAL_CKPT_FILE
        self.vocab_file = vocab_file or LOCAL_VOCAB_FILE
        self.ref_audio_path = ref_audio or F5_DEFAULT_REF_AUDIO
        self.ref_text = ref_text
        self.vocoder_name = vocoder_name
        self.speed = speed
        self.sample_rate = 24000

        # Kiểm tra file tồn tại
        if not os.path.exists(self.ckpt_file):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.ckpt_file}\n"
                f"Hãy đặt file model.pt vào: {MODEL_DIR}"
            )
        if not os.path.exists(self.vocab_file):
            raise FileNotFoundError(
                f"Vocab file not found: {self.vocab_file}\n"
                f"Hãy đặt file vocab.txt vào: {MODEL_DIR}"
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model và vocoder
        self._load_model()

    def _load_model(self):
        """Tải F5-TTS model, vocoder, và preprocess reference audio."""
        from omegaconf import OmegaConf

        # Load vocoder
        self.vocoder = load_vocoder(
            vocoder_name=self.vocoder_name,
            is_local=False,
            local_path="",
            device=self.device,
        )

        # Load model config
        model_cfg_path = os.path.join(
            F5_TTS_DIR, "src", "f5_tts", "configs", "F5TTS_Base.yaml"
        )
        model_cfg = OmegaConf.load(model_cfg_path).model

        # Load TTS model
        self.model = load_model(
            model_cls=DiT,
            model_cfg=model_cfg.arch,
            ckpt_path=self.ckpt_file,
            mel_spec_type=self.vocoder_name,
            vocab_file=self.vocab_file,
            device=self.device,
        )

        
    def synthesize(
        self,
        gen_text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Tổng hợp audio từ text sử dụng F5-TTS.

        Args:
            gen_text: Nội dung text cần tổng hợp thành giọng nói.
            ref_audio_path: Đường dẫn đến audio tham chiếu (voice cloning).
                Nếu None, sử dụng ref audio mặc định.
            ref_text: Transcript của ref audio.
                Nếu None, sử dụng ref_text mặc định.

        Returns:
            Tuple gồm:
                - numpy array chứa dữ liệu audio (float32)
                - sample rate (24000)
        """
        ref_audio_used = ref_audio_path or self.ref_audio_path
        ref_text_used = ref_text or self.ref_text

        # Lowercase text vì model được train với text lowercase
        gen_text_lower = gen_text.lower()
        ref_text_lower = ref_text_used.lower()

        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            ref_audio_used, ref_text_lower
        )
        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio=ref_audio_processed,
            ref_text=ref_text_processed,
            gen_text=gen_text_lower,
            model_obj=self.model,
            vocoder=self.vocoder,
            mel_spec_type=self.vocoder_name,
            speed=self.speed,
            device=self.device,
        )

        # Đảm bảo output là float32 numpy array
        if isinstance(audio_segment, torch.Tensor):
            audio_segment = audio_segment.cpu().numpy()

        audio_segment = audio_segment.astype(np.float32)

        return audio_segment, final_sample_rate

