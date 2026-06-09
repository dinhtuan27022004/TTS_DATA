#!/usr/bin/env bash

# Thiết lập GPU sử dụng
export CUDA_VISIBLE_DEVICES=0 # 0 nếu như bạn có GPU của nvidia :v

# Thêm thư mục src vào PYTHONPATH để tránh lỗi ModuleNotFoundError
export PYTHONPATH=src:$PYTHONPATH

log() {
    echo "$@"
}

# ---------------------------------------------------------------------------
# Parse arguments
# Sử dụng: bash fine_tuning.sh [--source_dir /path/to/dataset]
# Mặc định SOURCE_DIR sử dụng giá trị trong link_dataset.py nếu không truyền arg.
# ---------------------------------------------------------------------------
SOURCE_DIR="/workspace/TTS_DATA/data"
while [[ $# -gt 0 ]]; do
    case $1 in
        --source_dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Tạo thư mục cần thiết, 
DATASET_DIR="data/your_training_dataset"
mkdir -p "$DATASET_DIR"
# Bắt buộc phải có thư mục data/your_dataset chứa các file .wav, file .txt tương ứng, các bạn tự xử lý
mkdir -p data/your_dataset

# Định nghĩa các tham số huấn luyện
EXP_NAME="F5TTS_Base"
DATASET_NAME="your_training_dataset"
BATCH_SIZE=20000
NUM_WOKERS=64
WARMUP_UPDATES=5000
SAVE_UPDATES=10000
LAST_UPDATES=10000
EPOCHS=7 # <-- Bạn có thể chỉnh sửa số vòng lặp tại đây

# EXP_NAME="F5TTS_Base"
# DATASET_NAME="your_training_dataset"
# BATCH_SIZE=4000
# NUM_WOKERS=16
# WARMUP_UPDATES=20000
# SAVE_UPDATES=10000
# LAST_UPDATES=10000
# EPOCHS=10 # <-- Bạn có thể chỉnh sửa số vòng lặp tại đây
PRETRAIN_CKPT="ckpts/your_training_dataset/pretrained_model.pt"

# Tạo các biến stage để quản lý pipeline, bước nào đã chạy rồi thì không cần chạy lại
stage=5
stop_stage=5

# Link dữ liệu từ thư mục nguồn vào data/your_dataset bằng symlink
# Bước này LUÔN chạy trước tất cả các stage.
# Nếu muốn bỏ qua (đã link trước đó), hãy comment dòng bên dưới.
# log "Linking dataset from source dir into data/your_dataset ..."
# if [ -n "$SOURCE_DIR" ]; then
#     python link_dataset.py --source_dir "$SOURCE_DIR"
# else
#     python link_dataset.py
# fi


# Chuẩn hoá sample_rate, bỏ qua stage này nếu audio của bạn đã ở định dạng 24Khz
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Convert sample rate: data/your_dataset ..."
    python convert_sr.py
fi

# Chuẩn bị dữ liệu audio_name và text tương ứng
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Preparing metadata at: data/your_dataset ..."
    python prepare_metadata.py
fi

# Bổ sung từ vựng trong bộ dữ liệu của bạn chưa có trong từ vựng của mô hình pretrained
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Checking missing token in pretrained vocab ... "
    python check_vocab_pretrained.py
fi

# Mở rộng embedding của mô hình pretrained để hỗ trợ bộ từ vựng mới
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Extend embedding pretrained with new vocab ... "
    python extend_embedding_pretrained.py
fi

# Trích xuất đặc trưng
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Feature extraction ... "
    python src/f5_tts/train/datasets/prepare_csv_wavs.py "$DATASET_DIR" "$DATASET_DIR" --workers "$NUM_WOKERS"
fi

# Chạy quá trình fine-tuning
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Start fine-tuning F5-TTS with your dataset ... "
    python src/f5_tts/train/finetune_cli.py \
        --exp_name "$EXP_NAME" \
        --dataset_name "$DATASET_NAME" \
        --batch_size_per_gpu "$BATCH_SIZE" \
        --num_warmup_updates "$WARMUP_UPDATES" \
        --save_per_updates "$SAVE_UPDATES" \
        --last_per_updates "$LAST_UPDATES" \
        --finetune \
        --log_samples \
        --epochs "$EPOCHS" \
        --pretrain "$PRETRAIN_CKPT"
    ### Nếu bạn muốn training với nhiều gpu, sử dụng câu lệnh bên dưới:
    # accelerate launch src/f5_tts/train/finetune_cli.py \
    #     --exp_name "$EXP_NAME" \
    #     --dataset_name "$DATASET_NAME" \
    #     --batch_size_per_gpu "$BATCH_SIZE" \
    #     --num_warmup_updates "$WARMUP_UPDATES" \
    #     --save_per_updates "$SAVE_UPDATES" \
    #     --last_per_updates "$LAST_UPDATES" \
    #     --finetune \
    #     --log_samples \
    #     --epochs "$EPOCHS" \
    #     --pretrain "$PRETRAIN_CKPT"
fi

log "Fine-tuning F5-TTS done."
