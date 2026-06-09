"""Test so sánh F5-TTS giữa phiên bản Pretrained và Finetuned (model_last)."""

import os
import sys
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.tts.F5_V0 import F5TTSVietnamese

# 1. Định nghĩa tham số chung
gen_text = (
    "Xin chào, hôm nay là ngày 8 tháng 6 năm 2026. Tôi đang kiểm tra chất lượng của hệ thống chuyển văn bản thành giọng nói. "
    "Đây là một câu tiếng Việt bình thường để đánh giá độ tự nhiên, ngữ điệu và khả năng phát âm. "
    "Hệ thống cần đọc rõ ràng các dấu thanh như sắc, huyền, hỏi, ngã và nặng. "
    "Now let's switch to English. The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming the way people work, learn, and communicate around the world. "
    "Tiếp theo là phần trộn ngôn ngữ. Tôi đang sử dụng mô hình F5-TTS để tạo speech từ một đoạn text tiếng Việt, "
    "nhưng đôi khi cần đọc các từ tiếng Anh như machine learning, deep learning, speech synthesis, voice cloning và text-to-speech. "
    "Nhiệt độ ngoài trời hiện tại là 32 độ C. Tốc độ mạng đạt 150 megabit per second. "
    "Dung lượng tập dữ liệu khoảng 500 gigabytes, tương đương hơn nửa terabyte dữ liệu âm thanh. "
    "For a multilingual test, the speaker should be able to smoothly transition between Vietnamese and English "
    "without changing voice identity, speaking rate, or emotional style. Cảm ơn bạn đã lắng nghe. "
    "This is the end of the TTS evaluation sample."
)

ref_text = (
    "Mình là vì sao thế nhỉ, và chào mừng các bạn đã đến với chuỗi podcast "
    "thường bắt đầu bằng những câu hỏi vì sao dựa trên phương pháp first principle thinking."
)

ref_audio_path = "/workspace/TTS_DATA/hrehr.wav"


# 2. Khởi chạy mô hình Pretrained
print("\n=== [1/2] Đang khởi tạo mô hình Pretrained ===")
model_pretrain = F5TTSVietnamese(
    ckpt_file="/workspace/TTS_DATA/models/f5-tts-v0/model.pt",
    vocab_file="/workspace/TTS_DATA/models/f5-tts-v0/vocab.txt",
    vocoder_name="vocos",
    speed=1.0,
)

print("--- Đang tổng hợp giọng nói bằng Pretrained ---")
audio_pre, sr_pre = model_pretrain.synthesize(
    gen_text=gen_text,
    ref_text=ref_text,
    ref_audio_path=ref_audio_path
)
output_pre = "test_output_pretrained.wav"
sf.write(output_pre, audio_pre, sr_pre)
print(f"Lưu kết quả Pretrained tại: {output_pre} (samples={len(audio_pre)})\n")

# Giải phóng bộ nhớ GPU trước khi load model tiếp theo
del model_pretrain
import torch
torch.cuda.empty_cache()


# 3. Khởi chạy mô hình Finetuned (model_last)
print("=== [2/2] Đang khởi tạo mô hình Finetuned (model_last) ===")
model_last = F5TTSVietnamese(
    ckpt_file="/workspace/TTS_DATA/F5-TTS-Vietnamese/ckpts/your_training_dataset/model_last.pt",
    vocab_file="/workspace/TTS_DATA/F5-TTS-Vietnamese/data/your_training_dataset/vocab.txt",
    vocoder_name="vocos",
    speed=1.0,
)

print("--- Đang tổng hợp giọng nói bằng Finetuned ---")
audio_last, sr_last = model_last.synthesize(
    gen_text=gen_text,
    ref_text=ref_text,
    ref_audio_path=ref_audio_path
)
output_last = "test_output_last.wav"
sf.write(output_last, audio_last, sr_last)
print(f"Lưu kết quả Finetuned tại: {output_last} (samples={len(audio_last)})\n")

print("=== Hoàn thành so sánh cả 2 mô hình! ===")
