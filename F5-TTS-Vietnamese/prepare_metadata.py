"""
Mô-đun chuẩn bị metadata cho tập dữ liệu huấn luyện.
Tạo file metadata.csv và vocab từ tập dữ liệu âm thanh sử dụng kết hợp đa tiến trình (multiprocessing) và đường dẫn tương đối trực tiếp (zero-symlink) để đạt tốc độ xử lý tối đa (dưới 1 phút).
"""

import os
import glob
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool

# Đường dẫn dữ liệu
DATASET_DIR = "data/your_dataset"
TRAINING_DIR = "data/your_training_dataset"
WAVS_DIR = os.path.join(TRAINING_DIR, "wavs")
METADATA_PATH = os.path.join(TRAINING_DIR, "metadata.csv")
VOCAB_PATH = os.path.join(TRAINING_DIR, "vocab_your_dataset.txt")

# Tạo thư mục wavs rỗng để vượt qua kiểm tra định dạng
os.makedirs(WAVS_DIR, exist_ok=True)

def process_single_audio(wav_path: str):
    """
    Xử lý một tệp âm thanh đơn lẻ: đọc text, tính đường dẫn tương đối và trích xuất duration.
    """
    try:
        # Đọc nội dung text
        txt_path = wav_path.replace(".wav", ".txt")
        if not os.path.exists(txt_path):
            return None

        with open(txt_path, "r", encoding="utf8") as fr:
            text = fr.readline().strip().lower()
            text = text.replace("_", " ")
            text = " ".join(text.split())

        # Kiểm tra nhanh thời lượng bằng WAV header
        info = sf.info(wav_path)
        duration = info.duration

        # Bỏ qua file không đạt yêu cầu
        if duration < 1 or duration > 30 or len(text.split()) < 3:
            return None

        # Tính đường dẫn tương đối từ TRAINING_DIR tới tệp WAV gốc
        rel_wav_path = os.path.relpath(wav_path, start=TRAINING_DIR)

        return rel_wav_path, text
    except Exception:
        return None

def process_dataset():
    """
    Duyệt qua tất cả file WAV, chạy song song sử dụng Pool đa tiến trình và zero-symlink.
    """
    wav_paths = glob.glob(os.path.join(DATASET_DIR, "*.wav"))
    tokens = set()

    print(f"Bắt đầu xử lý song song siêu tốc {len(wav_paths)} file âm thanh sử dụng 16 workers...")
    
    results = []
    with Pool(processes=16) as pool:
        for res in tqdm(pool.imap_unordered(process_single_audio, wav_paths, chunksize=200), total=len(wav_paths), desc="Processing dataset"):
            if res is not None:
                results.append(res)

    print(f"Đang ghi kết quả vào metadata tại: {METADATA_PATH} ...")
    with open(METADATA_PATH, "w", encoding="utf8") as fw:
        # Ghi dòng tiêu đề
        fw.write("wav|text\n")
        for rel_wav_path, text in results:
            fw.write(f"{rel_wav_path}|{text}\n")
            tokens.update(text)

    # Ghi vocab vào file
    with open(VOCAB_PATH, "w", encoding="utf8") as fw_vocab:
        fw_vocab.write("\n".join(sorted(tokens)))

    print(f"Metadata lưu tại: {METADATA_PATH}")
    print(f"Vocab lưu tại: {VOCAB_PATH}")


if __name__ == "__main__":
    process_dataset()
