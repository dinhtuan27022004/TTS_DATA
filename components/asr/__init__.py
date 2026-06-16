"""ASR Model components."""

from components.asr.whisper_v3 import WhisperLargeV3ASR

_whisper_worker_instance = None

def get_whisper_worker(
    model_name: str = "large-v3",
    language: str = "vi",
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    num_workers: int = 1,
    word_timestamps: bool = False,
    use_batched_pipeline: bool = False,
):
    global _whisper_worker_instance
    if _whisper_worker_instance is None:
        _whisper_worker_instance = WhisperLargeV3ASR(
            model_name=model_name,
            language=language,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            num_workers=num_workers,
            word_timestamps=word_timestamps,
            use_batched_pipeline=use_batched_pipeline,
        )
    return _whisper_worker_instance

__all__ = ["WhisperLargeV3ASR", "get_whisper_worker"]
