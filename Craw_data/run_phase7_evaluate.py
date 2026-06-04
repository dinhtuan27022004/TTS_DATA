import os
import glob
import json
import re
import time
import logging
from tqdm import tqdm
import jiwer
from faster_whisper import WhisperModel

# --- CẤU HÌNH ---
INPUT_DIR = "Craw_data/Youtube_Data/Step_2"
RESULTS_FILE = "cer_results.json"
STATS_FILE = "cer_statistics.json"
SAVE_INTERVAL = 50  # Số lượng file xử lý trước khi lưu tạm (auto-save)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Chuyển thành chữ thường và xóa sạch dấu câu, chỉ giữ lại chữ/số và khoảng trắng."""
    if not text:
        return ""
    # Chuyển chữ thường
    text = text.lower()
    # Bỏ dấu câu (loại bỏ mọi thứ không phải là chữ cái, số, hoặc khoảng trắng)
    text = re.sub(r'[^\w\s]', '', text)
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_statistics(results: list):
    """Tính toán các chỉ số thống kê từ danh sách kết quả"""
    if not results:
        return {}
        
    cers = [item["cer"] for item in results]
    
    mean_cer = sum(cers) / len(cers)
    min_cer = min(cers)
    max_cer = max(cers)
    
    # Phân bổ CER
    distribution = {
        "Excellent (0-5%)": len([c for c in cers if c < 0.05]),
        "Good (5-10%)": len([c for c in cers if 0.05 <= c < 0.10]),
        "Acceptable (10-20%)": len([c for c in cers if 0.10 <= c < 0.20]),
        "Poor (>20%)": len([c for c in cers if c >= 0.20])
    }
    
    return {
        "total_files_evaluated": len(cers),
        "mean_cer": mean_cer,
        "min_cer": min_cer,
        "max_cer": max_cer,
        "distribution": distribution
    }

def main():
    print("=== Phase 7: Whisper CER Evaluation ===")
    
    # 1. Tìm tất cả file wav
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    wav_files.sort()
    
    if not wav_files:
        logger.error(f"Không tìm thấy file .wav nào trong {INPUT_DIR}")
        return
        
    logger.info(f"Tìm thấy {len(wav_files)} file .wav để xử lý.")
    
    # 2. Load kết quả cũ (Resume)
    processed_results = []
    processed_files = set()
    
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                processed_results = json.load(f)
                processed_files = {item["file"] for item in processed_results}
            logger.info(f"Đã load {len(processed_results)} kết quả từ lần chạy trước. Resume quá trình...")
        except Exception as e:
            logger.error(f"Lỗi khi đọc file kết quả cũ: {e}")
            
    # Lọc ra các file chưa xử lý
    files_to_process = [f for f in wav_files if os.path.basename(f) not in processed_files]
    logger.info(f"Số lượng file cần xử lý thêm: {len(files_to_process)}")
    
    if not files_to_process:
        logger.info("Đã đánh giá toàn bộ dataset. Cập nhật file thống kê...")
        stats = calculate_statistics(processed_results)
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        return

    # 3. Khởi tạo Whisper Model
    logger.info("Đang khởi tạo model Whisper (large-v3, int8)...")
    try:
        model = WhisperModel("large-v3", device="cuda", compute_type="int8")
    except Exception as e:
        logger.error(f"Lỗi khi tải model Whisper: {e}")
        return

    # 4. Chạy vòng lặp đánh giá
    new_results_count = 0
    
    for wav_path in tqdm(files_to_process, desc="Evaluating CER"):
        filename = os.path.basename(wav_path)
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        
        if not os.path.exists(txt_path):
            logger.warning(f"Không tìm thấy file .txt cho {filename}, bỏ qua.")
            continue
            
        # Đọc text Chunkformer (Reference)
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_raw = f.read()
            
        # Chạy Whisper (Hypothesis)
        try:
            segments, _ = model.transcribe(wav_path, beam_size=5, language="vi")
            hyp_raw = " ".join([seg.text for seg in segments])
        except Exception as e:
            logger.error(f"Lỗi khi transcribe file {filename}: {e}")
            continue
            
        # Normalize text
        ref_norm = normalize_text(ref_raw)
        hyp_norm = normalize_text(hyp_raw)
        
        # Nếu cả 2 chuỗi rỗng thì cho CER = 0, nếu chỉ 1 chuỗi rỗng thì CER = 1.0 (100%)
        if not ref_norm and not hyp_norm:
            cer_score = 0.0
        elif not ref_norm or not hyp_norm:
            cer_score = 1.0
        else:
            try:
                cer_score = jiwer.cer(ref_norm, hyp_norm)
            except Exception as e:
                logger.error(f"Lỗi khi tính CER cho {filename}: {e}")
                cer_score = 1.0
                
        # Lưu kết quả
        result_item = {
            "file": filename,
            "cer": round(cer_score, 4),
            "reference_text": ref_raw,
            "whisper_text": hyp_raw,
            "normalized_reference": ref_norm,
            "normalized_whisper": hyp_norm
        }
        processed_results.append(result_item)
        new_results_count += 1
        
        # Lưu Auto-save sau mỗi SAVE_INTERVAL file
        if new_results_count % SAVE_INTERVAL == 0:
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(processed_results, f, ensure_ascii=False, indent=2)
                
    # 5. Lưu toàn bộ kết quả cuối cùng khi vòng lặp hoàn tất
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)
        
    # 6. Tính toán và lưu Statistics
    stats = calculate_statistics(processed_results)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
        
    logger.info("Hoàn thành! Kết quả đã được lưu.")
    logger.info(f"Tổng hợp Statistics: {stats}")

if __name__ == "__main__":
    main()
