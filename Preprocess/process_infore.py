import os
import sys
import numpy as np
import soundfile as sf
import librosa
from datasets import load_from_disk
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Cấu hình đường dẫn
BASE_DIR = "/home/reg/TTS_DATA"
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "infore1_25hours")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "infore1_25hours_processed")

TARGET_SR = 24000
MIN_DURATION = 1.0
MAX_DURATION = 30.0

def process_single_example(args):
    idx, item = args
    try:
        audio = item.get("audio")
        transcription = item.get("transcription")
        
        if audio is None or transcription is None:
            return False, "missing_data"
            
        transcription = str(transcription).strip()
        if not transcription:
            return False, "empty_text"
            
        # Lấy mảng audio
        audio_array = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        duration = len(audio_array) / sr
        
        # Lọc độ dài
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            return False, "duration_filtered"
            
        # Chuẩn hóa âm thanh (resample nếu cần)
        if sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
            
        # Tạo tên file
        base_name = f"infore1_25hours_{idx:06d}"
        wav_path = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
        txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        
        # Ghi file wav và txt
        sf.write(wav_path, audio_array, TARGET_SR, subtype='PCM_16')
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcription)
            
        return True, None
    except Exception as e:
        return False, f"error: {str(e)}"

def main():
    print(f"Đang tải dataset từ: {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Lỗi: Không tìm thấy dataset tại {RAW_DATA_PATH}")
        sys.exit(1)
        
    dataset = load_from_disk(RAW_DATA_PATH)
    train_dataset = dataset["train"]
    total_examples = len(train_dataset)
    print(f"Tổng số mẫu cần xử lý: {total_examples}")
    
    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Thư mục output: {OUTPUT_DIR}")
    
    # Chuẩn bị tham số cho multiprocessing
    # Vì load_from_disk tải dạng lazy arrow dataset, ta có thể dùng chỉ số
    # để tránh việc chuyển toàn bộ dữ liệu qua IPC (processes).
    # Tuy nhiên, để tối ưu tốc độ đọc, ta có thể sinh map hoặc xử lý theo lô.
    # Để tránh pickle overhead của dataset object, ta truyền idx và ta load/slice
    # hoặc ta map bằng multiprocess map của Hugging Face datasets.
    # HF datasets có phương thức .map() chạy đa tiến trình cực kỳ tối ưu!
    
    print("Bắt đầu convert dataset sang định dạng chuẩn (wav+txt)...")
    
    # Định nghĩa hàm xử lý dùng cho dataset.map()
    def map_function(example, idx):
        audio = example["audio"]
        transcription = example["transcription"]
        
        if audio is not None and transcription is not None:
            transcription = str(transcription).strip()
            if transcription:
                audio_array = np.asarray(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                duration = len(audio_array) / sr
                
                if MIN_DURATION <= duration <= MAX_DURATION:
                    if sr != TARGET_SR:
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
                    
                    base_name = f"infore1_25hours_{idx:06d}"
                    wav_path = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
                    txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
                    
                    sf.write(wav_path, audio_array, TARGET_SR, subtype='PCM_16')
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(transcription)
                        
        return example

    # Chạy map với num_proc tương đương số CPU cores
    num_workers = min(cpu_count(), 16)
    print(f"Sử dụng {num_workers} workers để xử lý...")
    
    train_dataset.map(
        map_function,
        with_indices=True,
        num_proc=num_workers,
        desc="Converting dataset"
    )
    
    # Kiểm tra số lượng file được tạo ra
    generated_wavs = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]
    print(f"Xử lý hoàn tất! Đã tạo thành công {len(generated_wavs)} cặp file .wav và .txt tại {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
