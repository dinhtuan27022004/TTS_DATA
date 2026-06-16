#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script loại bỏ âm thanh nền/nhạc nền bằng Demucs cho toàn bộ các file WAV trong một thư mục đầu vào.
Ghi đè trực tiếp lên file gốc.
Cứ mỗi 5000 mẫu sẽ trích xuất 1 cặp âm thanh (gốc & sau xử lý) lưu vào /home/reg/TTS_DATA/TMP để nghe thử.
"""

import os
import sys
import logging
import shutil
import torch
import torchaudio
import threading
import warnings
from tqdm import tqdm

# Tắt các UserWarning (sinc_interpolation deprecation, v.v.) làm rối mắt và vỡ thanh tiến trình tqdm
warnings.filterwarnings("ignore", category=UserWarning)

# Cấu hình đường dẫn và mô hình mặc định
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "Processed_DATA", "PhoAudioBook")
MODEL_NAME = "htdemucs"
NUM_THREADS = 2   # Số lượng luồng xử lý song song

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TMP_DIR = "/home/reg/TTS_DATA/TMP"
STATE_FILE = os.path.join(BASE_DIR, "Processed_DATA", "preprocess_state.json")

import json
import tempfile
import datetime

def load_preprocess_state(state_file_path: str) -> dict:
    """Tải trạng thái xử lý từ file JSON."""
    if not os.path.exists(state_file_path):
        return {}
    try:
        with open(state_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        logger.error(f"Lỗi khi đọc file trạng thái {state_file_path}: {err}")
        return {}

def update_preprocess_state(state_file_path: str, key: str, last_file: str, success_count: int, error_count: int):
    """Cập nhật và ghi trạng thái xử lý vào file JSON một cách an toàn (atomic)."""
    state = load_preprocess_state(state_file_path)
    state[key] = {
        "last_processed_file": last_file,
        "success_count": success_count,
        "error_count": error_count,
        "updated_at": datetime.datetime.now().isoformat()
    }
    dir_name = os.path.dirname(state_file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp", encoding="utf-8") as tf:
            json.dump(state, tf, ensure_ascii=False, indent=4)
            temp_name = tf.name
        os.replace(temp_name, state_file_path)
    except Exception as err:
        logger.error(f"Lỗi khi ghi file trạng thái an toàn {state_file_path}: {err}")

class DemucsDenoiser:
    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
        self.model = None
        self.device = None
        self._load_model()

    def _load_model(self):
        """Load model Demucs và chọn device phù hợp."""
        from demucs.pretrained import get_model

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Sử dụng GPU (CUDA) cho Demucs")
        else:
            self.device = torch.device("cpu")
            logger.info("Sử dụng CPU cho Demucs")

        logger.info(f"Đang load model Demucs: {self.model_name}...")
        self.model = get_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model {self.model_name} đã load thành công!")

    def process_file(self, file_path: str, save_sample: bool = False) -> bool:
        """
        Xử lý loại bỏ nhạc nền cho một file WAV.
        Ghi đè trực tiếp lên file cũ.
        Nếu save_sample=True, lưu thêm bản sao gốc và bản sao đã xử lý vào TMP_DIR.
        """


        from demucs.apply import apply_model

        # Đọc file gốc
        wav, sr = torchaudio.load(file_path)
        orig_wav = wav.clone()  # Giữ lại bản clone với sample rate GỐC để lưu sample
        orig_sr = sr             # Lưu lại sample rate gốc trước khi resample

        # Demucs yêu cầu sample rate của model
        model_sr = self.model.samplerate
        if sr != model_sr:
            wav = torchaudio.functional.resample(wav, sr, model_sr)
            sr = model_sr

        # Đảm bảo stereo (2 channels)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]

        # Chạy qua model
        wav_batch = wav.unsqueeze(0).to(self.device)
        with torch.no_grad():
            sources = apply_model(self.model, wav_batch, device=self.device)

        # Lấy phần vocals (giọng nói sạch)
        vocals_idx = self.model.sources.index("vocals")
        vocals = sources[0, vocals_idx].cpu()  # (channels, samples)

        # Nếu cần lưu sample để nghe thử
        if save_sample:
            os.makedirs(TMP_DIR, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            orig_sample_path = os.path.join(TMP_DIR, f"{base_name}_original.wav")
            processed_sample_path = os.path.join(TMP_DIR, f"{base_name}_processed.wav")

            # Lưu bản GỐC với sample rate GỐC trước khi resample/xử lý
            torchaudio.save(orig_sample_path, orig_wav, orig_sr)
            # Lưu bản ĐÃ XỬ LÝ với sample rate của model
            torchaudio.save(processed_sample_path, vocals, sr)
            logger.info(f"[SAMPLE SAVED] Gốc ({orig_sr}Hz): {orig_sample_path}")
            logger.info(f"[SAMPLE SAVED] Đã xử lý ({sr}Hz): {processed_sample_path}")

        # Ghi đè trực tiếp lên file WAV gốc
        torchaudio.save(file_path, vocals, sr)
        return True


def main():
    input_dir = INPUT_DIR

    if not os.path.isdir(input_dir):
        logger.error(f"Thư mục đầu vào không tồn tại hoặc không phải là thư mục hợp lệ: {input_dir}")
        logger.info("Hãy kiểm tra biến 'INPUT_DIR' ở đầu file script.")
        sys.exit(1)

    denoiser = DemucsDenoiser(model_name=MODEL_NAME)

    # Đọc trạng thái đã lưu để khôi phục (Resume)
    state = load_preprocess_state(STATE_FILE)
    rb_state = state.get("remove_background", {})
    last_processed = rb_state.get("last_processed_file")
    success_count = rb_state.get("success_count", 0)
    error_count = rb_state.get("error_count", 0)
    processed_count = success_count + error_count

    if last_processed:
        logger.info(f"Phát hiện file trạng thái cũ. Khôi phục tiến trình từ file: {last_processed}")
        logger.info(f"Đã xử lý trước đó: Thành công {success_count}, Thất bại {error_count}")
    else:
        logger.info("Không tìm thấy trạng thái cũ hoặc bắt đầu mới từ đầu.")

    processed_files_set = set()

    while True:
        # Tìm tất cả các file WAV trong thư mục
        logger.info(f"Đang quét tìm file .wav trong {input_dir}...")
        all_wav_files = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    all_wav_files.append(os.path.join(root, f))

        # Sắp xếp toàn bộ danh sách để đảm bảo thứ tự bảng chữ cái tuyệt đối
        all_wav_files.sort()

        # Lọc ra các file chưa được xử lý (lớn hơn file đã lưu và chưa nằm trong set phiên chạy này)
        new_wav_files = []
        for filepath in all_wav_files:
            rel_path = os.path.relpath(filepath, input_dir)
            # Bỏ qua các file đã xử lý ở phiên trước (theo thứ tự alphabet)
            if last_processed and rel_path <= last_processed:
                continue
            # Bỏ qua các file đã xử lý ở phiên chạy hiện tại
            if filepath in processed_files_set:
                continue
            new_wav_files.append(filepath)

        total_new = len(new_wav_files)
        if total_new == 0:
            if len(processed_files_set) > 0 or last_processed:
                logger.info("Không phát hiện thêm file mới nào. Tiến trình hoàn tất.")
            else:
                logger.warning("Không tìm thấy file WAV nào. Kết thúc chương trình.")
            break

        logger.info(f"Phát hiện {total_new} file .wav mới/chưa xử lý (Tổng số file trong thư mục: {len(all_wav_files)}).")

        # Tách danh sách file thành NUM_THREADS phần
        thread_file_lists = [new_wav_files[i::NUM_THREADS] for i in range(NUM_THREADS)]

        lock = threading.Lock()
        completed_files = set()
        completed_prefix_idx = 0
        pbar = tqdm(total=total_new, desc="Processing audios")

        def worker_task(file_list, thread_name):
            nonlocal success_count, error_count, processed_count, completed_prefix_idx
            for filepath in file_list:
                with lock:
                    import random
                    save_sample = (random.randint(1, 5000) == 1)
                
                try:
                    denoiser.process_file(filepath, save_sample=save_sample)
                    with lock:
                        success_count += 1
                except Exception as err:
                    logger.error(f"[FAIL] [{thread_name}] Lỗi khi xử lý {filepath}: {err}")
                    with lock:
                        error_count += 1

                with lock:
                    processed_files_set.add(filepath)
                    completed_files.add(filepath)
                    processed_count += 1

                    # Di chuyển con trỏ prefix để tìm file liên tục lớn nhất đã hoàn thành
                    while (completed_prefix_idx < len(new_wav_files) and 
                           new_wav_files[completed_prefix_idx] in completed_files):
                        completed_prefix_idx += 1

                    # Cập nhật trạng thái an toàn dựa trên file liên tục lớn nhất đã hoàn thành
                    if completed_prefix_idx > 0:
                        last_consecutive_file = new_wav_files[completed_prefix_idx - 1]
                        rel_path = os.path.relpath(last_consecutive_file, input_dir)
                        update_preprocess_state(
                            state_file_path=STATE_FILE,
                            key="remove_background",
                            last_file=rel_path,
                            success_count=success_count,
                            error_count=error_count
                        )

                    pbar.update(1)

        # Khởi chạy NUM_THREADS thread song song
        threads = []
        for i in range(NUM_THREADS):
            t = threading.Thread(
                target=worker_task,
                args=(thread_file_lists[i], f"Thread-{i}"),
                daemon=True
            )
            threads.append(t)
            t.start()

        # Join các thread với timeout để cho phép nhận tín hiệu Ctrl+C (KeyboardInterrupt)
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.1)
        except KeyboardInterrupt:
            logger.warning("Bắt được tín hiệu hủy lệnh (Ctrl+C). Đang dừng tiến trình...")
            sys.exit(1)

        pbar.close()

    logger.info("=== KẾT QUẢ XỬ LÝ ===")
    logger.info(f"Tổng số file đã xử lý trong phiên này: {len(processed_files_set)}")
    logger.info(f"Tổng số file đã hoàn thành từ trước đến nay: {success_count + error_count} (Thành công: {success_count}, Thất bại: {error_count})")
    logger.info(f"Các mẫu nghe so sánh đã được lưu tại: {TMP_DIR}")

if __name__ == "__main__":
    main()
