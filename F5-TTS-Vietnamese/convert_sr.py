import glob
import os
from multiprocessing import Pool
from pathlib import Path
from shutil import move

import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm


def convert_and_replace(audio_path: str) -> None:
    """
    Xử lý atomic một file âm thanh:
      1. Bỏ qua nếu đã là file _24k.wav (tránh double-process)
      2. Convert sang 24kHz và ghi ra file tạm _24k.wav
      3. Xóa file/symlink gốc
      4. Đổi tên file tạm về tên gốc
    Toàn bộ 3 bước trên thực hiện trong 1 worker → không có race condition.
    """
    if "_24k.wav" in audio_path:
        return

    audio_path = Path(audio_path)
    output_path = audio_path.with_name(f"{audio_path.stem}_24k.wav")

    # Đọc file bằng soundfile
    data, sample_rate = sf.read(str(audio_path))

    # Chuyển đổi sang torch tensor
    if len(data.shape) == 1:
        waveform = torch.tensor(data).float().unsqueeze(0)
    else:
        waveform = torch.tensor(data).float().T

    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
        waveform = resampler(waveform)

    # Chuyển đổi sang mono nếu là stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Ghi file tạm _24k.wav
    out_data = waveform.squeeze().numpy()
    sf.write(str(output_path), out_data, 24000, subtype='PCM_16')

    # Xóa file/symlink gốc (os.remove chỉ xóa symlink, không xóa file thật)
    os.remove(str(audio_path))

    # Đổi tên file tạm về tên gốc
    move(str(output_path), str(audio_path))


if __name__ == "__main__":
    dataset_path = "data/your_dataset/*.wav"

    wav_paths = glob.glob(dataset_path)
    print(f"Tìm thấy {len(wav_paths)} file WAV, bắt đầu convert song song ...")

    with Pool(processes=16) as pool:
        list(tqdm(pool.imap_unordered(convert_and_replace, wav_paths, chunksize=500),
                  total=len(wav_paths), desc="Converting sample rate"))

    print("Hoàn thành convert sample rate.")