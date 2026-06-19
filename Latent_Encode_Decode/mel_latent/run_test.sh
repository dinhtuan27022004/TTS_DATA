#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_test.sh — Tự động tìm checkpoint và file wav để chạy test reconstruction
# ─────────────────────────────────────────────────────────────────────────────

# Thiết lập thư mục gốc của dự án
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Tìm file checkpoint (lấy file mới nhất được tạo)
CHECKPOINT=$(ls -t checkpoints/*.pt 2>/dev/null | head -n 1)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ Lỗi: Không tìm thấy bất kỳ checkpoint (.pt) nào trong thư mục 'checkpoints/'."
    echo "Hãy chạy train trước hoặc đảm bảo bạn đã lưu checkpoint."
    exit 1
fi

# 2. Tìm file WAV để test
# Nếu người dùng truyền đường dẫn file wav khi chạy script (ví dụ: ./run_test.sh test.wav) thì dùng file đó
WAV_PATH="$1"
if [ -z "$WAV_PATH" ]; then
    # Tự động quét file .wav đầu tiên tìm thấy trong thư mục data/wavs/
    WAV_PATH=$(find data/wavs/ -name "*.wav" | head -n 1)
fi

if [ -z "$WAV_PATH" ] || [ ! -f "$WAV_PATH" ]; then
    echo "❌ Lỗi: Không tìm thấy file âm thanh .wav nào để chạy thử nghiệm."
    echo "Vui lòng đặt ít nhất một file .wav vào thư mục 'data/wavs/' hoặc truyền trực tiếp đường dẫn file:"
    echo "  bash run_test.sh path/to/your/audio.wav"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Đang chạy Test Reconstruction..."
echo "   Checkpoint: $CHECKPOINT"
echo "   File WAV:   $WAV_PATH"
echo "   Đầu ra:     output_reconstruct/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 3. Chạy lệnh python test
python3 src/inference_reconstruct.py \
    --config configs/config.yaml \
    --checkpoint "$CHECKPOINT" \
    --wav_path "$WAV_PATH" \
    --out_dir output_reconstruct \
    --plot

if [ $? -eq 0 ]; then
    echo "✅ Hoàn thành! Kết quả so sánh đã được lưu tại: output_reconstruct/reconstruction_comparison.png"
else
    echo "❌ Có lỗi xảy ra trong quá trình chạy inference."
fi
