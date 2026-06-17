import logging
import torch
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

_classifier = None

def _load_classifier():
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SpeechBrain Speaker Recognition model (ECAPA-TDNN)...")
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        logger.info("SpeechBrain Speaker Recognition model loaded.")
    return _classifier

def _get_embedding(audio: np.ndarray, sr: int) -> torch.Tensor:
    classifier = _load_classifier()
    
    # Convert numpy array to torch tensor
    tensor = torch.from_numpy(audio).unsqueeze(0) # shape: (1, samples)
    
    # Resample to 16000 Hz if needed
    if sr != 16000:
        import torchaudio.functional as F
        tensor = F.resample(tensor, sr, 16000)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = tensor.to(device)
    
    with torch.no_grad():
        embeddings = classifier.encode_batch(tensor)
        
    return embeddings.squeeze(0).squeeze(0) # shape: (192,) or similar

def compute_speaker_similarity(ref_audio: np.ndarray, syn_audio: np.ndarray, sr: int) -> Optional[float]:
    """Tính cosine similarity của speaker embedding giữa ref_audio và syn_audio."""
    try:
        emb_ref = _get_embedding(ref_audio, sr)
        emb_syn = _get_embedding(syn_audio, sr)
        
        sim = torch.nn.functional.cosine_similarity(emb_ref, emb_syn, dim=-1)
        return float(sim.item())
    except Exception as exc:
        logger.warning("Không thể tính Speaker Similarity: %s", exc)
        return None
