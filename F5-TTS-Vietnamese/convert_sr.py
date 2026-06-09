import glob
import os
from multiprocessing import Pool
from pathlib import Path
from shutil import move

import soundfile as sf
import soxr
from tqdm import tqdm


def convert_and_replace(audio_path: str) -> None:
    """
    Xử lý atomic một file âm thanh:
      1. Bỏ qua nếu đã là file _24k.wav (tránh double-process)
      2. Convert sang 24kHz sử dụng soxr và ghi ra file tạm _24k.wav
      3. Xóa file/symlink gốc
      4. Đổi tên file tạm về tên gốc
    """
    if "_24k.wav" in audio_path:
        return

    audio_path = Path(audio_path)
    output_path = audio_path.with_name(f"{audio_path.stem}_24k.wav")

    try:
        # Đọc file bằng soundfile
        data, sample_rate = sf.read(str(audio_path))

        # Chuyển đổi sang mono nếu là stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Resample sang 24kHz nếu sample_rate khác 24000
        if sample_rate != 24000:
            data = soxr.resample(data, sample_rate, 24000)

        # Ghi file tạm _24k.wav
        sf.write(str(output_path), data, 24000, subtype='PCM_16')

        # Xóa file/symlink gốc (os.remove chỉ xóa symlink, không xóa file thật)
        os.remove(str(audio_path))

        # Đổi tên file tạm về tên gốc
        move(str(output_path), str(audio_path))
    except Exception as e:
        print(f"\nLỗi khi xử lý file {audio_path}: {e}")


if __name__ == "__main__":
    dataset_path = "data/your_dataset/*.wav"

    wav_paths = glob.glob(dataset_path)
    num_files = len(wav_paths)
    print(f"Tìm thấy {num_files} file WAV, bắt đầu convert song song ...")

    if num_files > 0:
        # Máy của bạn có rất nhiều CPU (224 cores), thiết lập số process phù hợp
        # Cập nhật số processes lên cao hơn (ví dụ: 64) để tận dụng phần cứng
        num_workers = min(64, os.cpu_count() or 16)
        print(f"Sử dụng {num_workers} processes để convert song song...")
        
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(convert_and_replace, wav_paths, chunksize=500),
                      total=num_files, desc="Converting sample rate"))

    print("Hoàn thành convert sample rate.")