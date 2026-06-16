import os
import sys
import glob
import logging
import time
import gc
import torch
from tqdm import tqdm
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- CẤU HÌNH ---
INPUT_DIR = "Craw_data/Youtube_Data/Step_2"
SKIP_UNTIL_FILE = ""  # Điền tên file .wav để bỏ qua từ đầu đến file này (kể cả nó). VD: "abc.wav"
QUEUE_BATCH_SIZE = 16
QUEUE_TIMEOUT = 0.2
WHISPER_BATCH_SIZE = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():

    print("=== Phase 6: Whisper Transcription ===")
    
    # 1. Khởi tạo Whisper Model (Chỉ 1 lần ngoài vòng lặp)
    logger.info("Đang khởi tạo model Whisper (large-v3, float16) với 4 workers...")
    try:
        model = WhisperModel("large-v3", device="cuda", compute_type="float16", num_workers=1)
    except Exception as e:
        logger.error(f"Lỗi khi tải model Whisper: {e}")
        return

    def process_file(wav_path):
        filename = os.path.basename(wav_path)
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
            
        try:
            # Chạy Whisper để nhận diện giọng nói
            segments, _ = model.transcribe(wav_path, beam_size=5, language="vi")
            
            # Ghép các đoạn text lại với nhau (bỏ qua các khoảng trắng thừa)
            text_content = " ".join([seg.text.strip() for seg in segments]).strip()
            
            # Ghi ra file .txt
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_content)
                
        except Exception as e:
            logger.error(f"Lỗi khi transcribe file {filename}: {e}")

    # 2. Vòng lặp liên tục
    while True:
        wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
        wav_files.sort()
        
        if SKIP_UNTIL_FILE:
            skip_target = os.path.basename(SKIP_UNTIL_FILE)
            try:
                target_idx = next(i for i, f in enumerate(wav_files) if os.path.basename(f) == skip_target)
                wav_files = wav_files[target_idx + 1:]
            except StopIteration:
                pass
        
        # Lọc ra các file chưa có file .txt (Resume support)
        files_to_process = []
        for wav_path in wav_files:
            txt_path = os.path.splitext(wav_path)[0] + ".txt"
            if not os.path.exists(txt_path):
                files_to_process.append(wav_path)
                
        if not files_to_process:
            logger.info("Chưa có file wav mới nào. Chờ 10 giây và thử lại...")
            time.sleep(10)
            continue

        logger.info(f"Số lượng file CẦN xử lý thêm lô này: {len(files_to_process)}")

        processed_count = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_file, p): p for p in files_to_process}
            for _ in tqdm(as_completed(futures), total=len(files_to_process), desc="Transcribing"):
                processed_count += 1
                if processed_count % 1000 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
        logger.info(f"Hoàn thành xử lý {len(files_to_process)} files trong lô vừa rồi.")

if __name__ == "__main__":
    main()
