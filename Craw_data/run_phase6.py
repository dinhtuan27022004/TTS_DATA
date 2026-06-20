import os
import sys
import glob
import json
import logging
import time
import gc
import torch
import wave
from tqdm import tqdm
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- CẤU HÌNH ---
INPUT_DIR = "/home/reg/TTS_DATA/Craw_data/Youtube_Data/Step_2"
SKIP_UNTIL_FILE = ""  # Điền tên file .wav để bỏ qua từ đầu đến file này (kể cả nó). VD: "abc.wav"
QUEUE_BATCH_SIZE = 16
QUEUE_TIMEOUT = 0.2
WHISPER_BATCH_SIZE = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
_chunkformer_model = None

def get_chunkformer_model():
    global _chunkformer_model
    if _chunkformer_model is None:
        logger.info("Đang khởi tạo model ChunkFormer (khanhld/chunkformer-ctc-large-vie) làm fallback...")
        from chunkformer import ChunkFormerModel
        _chunkformer_model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie").to("cuda")
    return _chunkformer_model

def get_audio_duration(wav_path):
    try:
        with wave.open(wav_path, "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception as e:
        logger.error(f"Lỗi khi đọc độ dài file wav {wav_path}: {e}")
        return 0.0

def has_repetition(text, min_len=8):
    if not text:
        return False
    words = text.strip().lower().split()
    n = len(words)
    seen = set()
    for i in range(n - min_len + 1):
        ngram = tuple(words[i:i+min_len])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False

def is_anomalous_ws(text, duration):
    if duration <= 0:
        return True
    words = text.strip().split()
    w_s = len(words) / duration
    return w_s > 8.0 or w_s < 1.0

def is_hallucinated_outro(text):
    if not text:
        return False
    # Clean text
    cleaned = text.strip().lower()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1].strip()
    if cleaned.endswith('.'):
        cleaned = cleaned[:-1].strip()
        
    # Check against target phrases
    target_phrases = {
        "các bạn hãy đăng ký kênh để ủng hộ kênh của mình nhé",
        "hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn",
        "hãy subscribe cho kênh la la school để không bỏ lỡ những video hấp dẫn",
        "hãy đăng ký kênh để ủng hộ kênh của mình nhé"
    }
    
    # Exact match or substring containment
    for target in target_phrases:
        if cleaned == target or target in cleaned:
            return True
            
    # Also check if it matches common outro keywords
    outro_keywords = ["đăng ký kênh để ủng hộ", "subscribe cho kênh"]
    for kw in outro_keywords:
        if kw in cleaned:
            return True
            
    return False

def build_word_timestamps_from_whisper(segments):
    words = []
    segment_items = []
    for seg in segments:
        segment_items.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
        })
        for word in seg.words or []:
            item = {
                "word": word.word.strip(),
                "start": round(float(word.start), 3),
                "end": round(float(word.end), 3),
            }
            if getattr(word, "probability", None) is not None:
                item["probability"] = round(float(word.probability), 4)
            if item["word"]:
                words.append(item)
    return words, segment_items

def build_approx_word_timestamps(text, duration):
    tokens = text.strip().split()
    if not tokens or duration <= 0:
        return []
    step = duration / len(tokens)
    return [
        {
            "word": word,
            "start": round(i * step, 3),
            "end": round((i + 1) * step, 3),
            "source": "estimated",
        }
        for i, word in enumerate(tokens)
    ]



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
            # Kiểm tra thời lượng file wav trước khi chạy Whisper
            duration = get_audio_duration(wav_path)
            if duration < 0.3 or duration > 30:
                logger.warning(f"File {filename} có thời lượng không hợp lệ ({duration:.2f}s). Tiến hành XÓA file.")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                return

            # Chạy Whisper để nhận diện giọng nói
            segments, _ = model.transcribe(wav_path, beam_size=5, language="vi", word_timestamps=True)
            segments = list(segments)
            
            # Ghép các đoạn text lại với nhau (bỏ qua các khoảng trắng thừa)
            text_content = " ".join([seg.text.strip() for seg in segments]).strip()
            
            # Kiểm tra thời lượng file wav
            duration = get_audio_duration(wav_path)
            
            # Kiểm tra xem có bị lỗi ảo giác outro, lỗi lặp từ, hoặc w/s dị thường không
            has_error = (
                is_hallucinated_outro(text_content) or 
                has_repetition(text_content) or 
                is_anomalous_ws(text_content, duration)
            )
            
            if has_error:
                current_ws = len(text_content.split()) / duration if duration > 0 else 0.0
                logger.warning(f"Phát hiện lỗi/dị thường trong Whisper cho {filename} (w/s={current_ws:.2f}): '{text_content}'")
                logger.info(f"Đang chạy fallback qua ChunkFormer cho file {filename}...")
                try:
                    cf_model = get_chunkformer_model()
                    cf_text = cf_model.endless_decode(wav_path, return_timestamps=False)
                    logger.info(f"Kết quả ChunkFormer: '{cf_text}'")
                    
                    # Sau khi fallback qua ChunkFormer, kiểm tra lại xem có còn bị lỗi không
                    has_error_cf = (
                        is_hallucinated_outro(cf_text) or 
                        has_repetition(cf_text) or 
                        is_anomalous_ws(cf_text, duration)
                    )
                    
                    if has_error_cf:
                        cf_ws = len(cf_text.split()) / duration if duration > 0 else 0.0
                        logger.error(f"Sau khi fallback ChunkFormer, file {filename} vẫn bị lỗi/dị thường (w/s={cf_ws:.2f}): '{cf_text}'. Tiến hành XÓA file.")
                        # Xóa cả file wav và file txt nếu có tồn tại
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                        if os.path.exists(txt_path):
                            os.remove(txt_path)
                        return
                    else:
                        text_content = cf_text
                except Exception as e:
                    logger.error(f"Lỗi khi chạy fallback ChunkFormer cho {filename}: {e}. Tiến hành XÓA file.")
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    return
            
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
        for wav_path in tqdm(files_to_process, desc="Transcribing"):
            process_file(wav_path)
            processed_count += 1
            if processed_count % 1000 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        logger.info(f"Hoàn thành xử lý {len(files_to_process)} files trong lô vừa rồi.")

if __name__ == "__main__":
    main()
