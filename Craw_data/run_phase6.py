"""
Phase 6: Automatic Punctuation Restoration
Script chạy Phase 6 - Thêm dấu tự động cho file JSON trong Step 1 và kiểm tra tính hợp lệ với timestamps.

Cách dùng:
    cd D:\CO_2026\TTS-DATA  (hoặc cd tới thư mục gốc của project)
    python Craw_data/run_phase6.py
"""

import sys
import os
import json
import re
import logging
from tqdm import tqdm
from pathlib import Path
import unicodedata

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PunctuationRestorer:
    """
    Module phụ trách thêm dấu tự động cho văn bản.
    Sử dụng model tuananh18/VietnamesePunctuation để predict nhãn dấu câu.
    """
    def __init__(self):
        self.use_transformers = False
        try:
            from transformers import pipeline
            logger.info("Đang load model punctuation (tuananh18/VietnamesePunctuation)...")
            # aggregation_strategy=None để lấy offset chính xác của từng token
            self.pipe = pipeline(
                "token-classification",
                model="tuananh18/VietnamesePunctuation",
                aggregation_strategy=None
            )
            self.use_transformers = True
        except ImportError:
            logger.warning("Không tìm thấy thư viện transformers, sử dụng phương thức tạm thời.")

    @staticmethod
    def _entity_to_punct(entity: str) -> str:
        """Chuyển nhãn model sang ký tự dấu câu tương ứng."""
        e = entity.upper()
        if 'COMMA' in e or e == 'LABEL_1':
            return ','
        if 'PERIOD' in e or e == 'LABEL_2':
            return '.'
        if 'QUESTION' in e or e == 'LABEL_3':
            return '?'
        if 'EXCLAMATION' in e or e == 'LABEL_4':
            return '!'
        if 'COLON' in e or e == 'LABEL_5':
            return ':'
        if 'SEMICOLON' in e or e == 'LABEL_6':
            return ';'
        return ''

    def restore_with_originals(self, original_words: list) -> list:
        """
        Nhận danh sách từ gốc (có dấu tiếng Việt), predict dấu câu,
        trả về danh sách từ gốc đã được gắn thêm dấu câu.
        Đây là phương thức chính để tránh mất dấu tiếng Việt.
        """
        if not self.use_transformers:
            return original_words  # fallback: giữ nguyên

        # Tạo text bóc dấu câu cũ VÀ bóc dấu tiếng Việt
        # (model được train trên text ASCII không dấu)
        clean_words_viet = [re.sub(r'[^\w\s]', '', w) for w in original_words]  # bóc dấu câu, GIỮ dấu TV
        clean_words_ascii = [
            unicodedata.normalize('NFD', w)
            .encode('ascii', 'ignore')
            .decode('ascii')
            .strip()
            for w in clean_words_viet
        ]  # bóc luôn dấu tiếng Việt → ASCII
        clean_text = ' '.join(clean_words_ascii)

        try:
            # Model predict trên text ASCII không có dấu câu
            outputs = self.pipe(clean_text)

            # Xây dựng bảng: word_index -> nhãn dấu câu
            # Dùng char offset trong clean_text (ASCII) để xác định từ thứ mấy
            word_end_offsets = []
            pos = 0
            for w in clean_words_ascii:
                pos += len(w)
                word_end_offsets.append(pos)  # char index ngay sau từ trong ASCII text
                pos += 1  # khoảng trắng

            # punct_for_word[i] = dấu câu gắn sau từ thứ i
            punct_for_word = [''] * len(clean_words_viet)

            for token in outputs:
                entity = token.get('entity', 'O')
                if entity == 'O':
                    continue
                punct = self._entity_to_punct(entity)
                if not punct:
                    continue

                # Tìm từ tương ứng dựa trên token['end'] (char offset kết thúc trong ASCII text)
                token_end = token.get('end', -1)
                for i, end_off in enumerate(word_end_offsets):
                    if token_end <= end_off:
                        # Ưu tiên dấu câu mạnh hơn nếu đã có
                        existing = punct_for_word[i]
                        if not existing or '.' in punct or '?' in punct or '!' in punct:
                            punct_for_word[i] = punct
                        break

            # Ghép từ tiếng Việt GỐC (có dấu) + dấu câu được predict
            result = [clean_words_viet[i] + punct_for_word[i] for i in range(len(clean_words_viet))]
            return result

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán dấu câu: {e}")
            return original_words  # fallback: giữ nguyên

    def restore(self, text: str) -> str:
        """
        Legacy method: nhận text string, trả về text string có dấu câu.
        Không còn được dùng trong pipeline chính.
        """
        return text



def process_file(json_path: str, restorer: PunctuationRestorer) -> bool:
    """Xử lý thêm dấu cho một file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"[SKIP] Lỗi định dạng JSON ở file {os.path.basename(json_path)}: {e}")
        return False
    except Exception as e:
        logger.error(f"[SKIP] Lỗi khi đọc file {os.path.basename(json_path)}: {e}")
        return False
        
    word_timestamps = data.get('word_timestamps', [])
    if not word_timestamps:
        logger.warning(f"[SKIP] Không có word_timestamps trong {json_path}")
        return False
        
    # Bước 1: Trích xuất danh sách từ gốc (giữ nguyên dấu tiếng Việt)
    original_words = [wt['word'] for wt in word_timestamps]
    
    # Bước 2: Model AI predict dấu câu, trả thẳng về list từ đã gắn dấu câu
    # Phương thức này KHÔNG thay đổi ký tự tiếng Việt, chỉ thêm dấu câu như , . ? ! vào cuối từ
    aligned_words = restorer.restore_with_originals(original_words)

    # Bước 3: Sanity check - đảm bảo số lượng từ không thay đổi
    if len(aligned_words) != len(original_words):
        logger.error(f"[FAIL] {os.path.basename(json_path)}: Số lượng từ thay đổi sau khi thêm dấu ({len(original_words)} -> {len(aligned_words)})")
        return False
        
    # Bước 4: Cập nhật file JSON với dữ liệu mới
    for i in range(len(word_timestamps)):
        word_timestamps[i]['word'] = aligned_words[i]
        
    data['word_timestamps'] = word_timestamps
    # Tạo lại full_text dựa trên kết quả mới ghép lại
    data['full_text'] = " ".join(aligned_words)
    
    # Ghi đè hoặc lưu ra file (ở đây mặc định ghi đè, bạn có thể chỉnh lưu sang thư mục khác nếu muốn)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    return True

def main():
    print("=== Phase 6: Automatic Punctuation Restoration ===")
    input_dir = "Craw_data/Youtube_Data/Step_1"
    
    if not os.path.exists(input_dir):
        logger.error(f"Thư mục {input_dir} không tồn tại!")
        return

    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    json_files.sort()
    
    if not json_files:
        logger.info("Không tìm thấy file JSON nào.")
        return

    logger.info(f"Tìm thấy {len(json_files)} file JSON. Đang nạp model...")
    restorer = PunctuationRestorer()
    
    success_count = 0
    fail_count = 0
    
    for filename in tqdm(json_files, desc="Processing"):
        filepath = os.path.join(input_dir, filename)
        if process_file(filepath, restorer):
            success_count += 1
        else:
            fail_count += 1
            
    print("\n=== Hoàn thành ===")
    print(f"Thành công: {success_count} files")
    print(f"Thất bại:   {fail_count} files")

if __name__ == "__main__":
    main()
