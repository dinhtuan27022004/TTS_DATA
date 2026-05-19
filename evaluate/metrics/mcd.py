"""
Mel Cepstral Distortion (MCD) metric.

Tính toán khoảng cách Euclidean giữa mel cepstral coefficients
của audio tham chiếu và audio tổng hợp.
"""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def compute_mcd(ref_audio: np.ndarray, syn_audio: np.ndarray, sr: int) -> float:
    """Tính Mel Cepstral Distortion giữa audio tham chiếu và audio tổng hợp.

    MCD đo khoảng cách Euclidean trung bình giữa các vector MFCC
    của hai tín hiệu audio. Giá trị thấp hơn cho thấy chất lượng tốt hơn.

    Args:
        ref_audio: Audio tham chiếu dạng numpy array (float32).
        syn_audio: Audio tổng hợp dạng numpy array (float32).
        sr: Sample rate của cả hai tín hiệu.

    Returns:
        Giá trị MCD dạng float không âm (đơn vị dB).
    """
    # Căn chỉnh độ dài: pad tín hiệu ngắn hơn bằng zero
    ref_len = len(ref_audio)
    syn_len = len(syn_audio)

    if ref_len > syn_len:
        syn_audio = np.pad(syn_audio, (0, ref_len - syn_len), mode="constant")
    elif syn_len > ref_len:
        ref_audio = np.pad(ref_audio, (0, syn_len - ref_len), mode="constant")

    # Trích xuất MFCC (13 coefficients, bỏ coefficient 0)
    n_mfcc = 13
    ref_mfcc = librosa.feature.mfcc(y=ref_audio.astype(np.float32), sr=sr, n_mfcc=n_mfcc + 1)[1:]
    syn_mfcc = librosa.feature.mfcc(y=syn_audio.astype(np.float32), sr=sr, n_mfcc=n_mfcc + 1)[1:]

    # Căn chỉnh số frame nếu khác nhau
    min_frames = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_frames]
    syn_mfcc = syn_mfcc[:, :min_frames]

    # Tính khoảng cách Euclidean trung bình giữa các frame
    diff = ref_mfcc - syn_mfcc
    frame_distances = np.sqrt(np.sum(diff**2, axis=0))
    mcd = float(np.mean(frame_distances))

    # Hệ số chuyển đổi sang dB (10 * sqrt(2) / ln(10))
    mcd_db = (10.0 * np.sqrt(2.0) / np.log(10.0)) * mcd

    logger.debug("MCD computed: %.4f dB", mcd_db)
    return mcd_db
