import os
import glob
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CẤU HÌNH ---
INPUT_DIR = "Craw_data/Youtube_Data/Step_2"
SKIP_UNTIL_FILE = ""  # Điền tên file .wav để bỏ qua từ đầu đến file này (kể cả nó). VD: "abc.wav"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():

    print("=== Phase 8: Whisper Transcription ===")
    
    # 1. Tìm tất cả file wav
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    wav_files.sort()
    
    if not wav_files:
        logger.error(f"Không tìm thấy file .wav nào trong {INPUT_DIR}")
        return
        
    logger.info(f"Tổng số gốc: {len(wav_files)} file .wav.")
    
    if SKIP_UNTIL_FILE:
        skip_target = os.path.basename(SKIP_UNTIL_FILE)
        try:
            target_idx = next(i for i, f in enumerate(wav_files) if os.path.basename(f) == skip_target)
            wav_files = wav_files[target_idx + 1:]
            logger.info(f"Đã bỏ qua các file từ đầu đến '{skip_target}'. Danh sách rút gọn còn {len(wav_files)} file.")
        except StopIteration:
            logger.warning(f"Không tìm thấy file '{skip_target}' trong danh sách. Không bỏ qua file nào.")
    
    # 2. Lọc ra các file chưa có file .txt (Resume support)
    files_to_process = []
    for wav_path in wav_files:
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            files_to_process.append(wav_path)
            
    logger.info(f"Số lượng file ĐÃ xử lý (có sẵn .txt): {len(wav_files) - len(files_to_process)}")
    logger.info(f"Số lượng file CẦN xử lý thêm: {len(files_to_process)}")
    
    if not files_to_process:
        logger.info("Đã transcribe toàn bộ dataset. Không cần chạy thêm.")
        return

    # 3. Khởi tạo Whisper Model
    logger.info("Đang khởi tạo model Whisper (large-v3, int8) với 2 workers...")
    try:
        model = WhisperModel("large-v3", device="cuda", compute_type="int8", num_workers=4)
    except Exception as e:
        logger.error(f"Lỗi khi tải model Whisper: {e}")
        return

    # 4. Chạy vòng lặp Transcribe
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

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, p): p for p in files_to_process}
        for _ in tqdm(as_completed(futures), total=len(files_to_process), desc="Transcribing"):
            pass
            
    logger.info("Hoàn thành quá trình Transcribe toàn bộ dataset!")

if __name__ == "__main__":
    main()
