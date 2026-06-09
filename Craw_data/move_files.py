import os
import shutil

d = '/home/reg/TTS_DATA/Craw_data/Youtube_Data/Step_2'
backup_d = '/home/reg/TTS_DATA/Craw_data/Youtube_Data/step2_backup'
os.makedirs(backup_d, exist_ok=True)

# Lấy danh sách các file wav và txt
wavs = set(f[:-4] for f in os.listdir(d) if f.endswith('.wav'))
txts = set(f[:-4] for f in os.listdir(d) if f.endswith('.txt'))

# Các file chưa có txt
missing = wavs - txts

moved = 0
for base in missing:
    src = os.path.join(d, base + '.wav')
    dst = os.path.join(backup_d, base + '.wav')
    try:
        os.rename(src, dst)
        moved += 1
    except Exception as e:
        print(f"Lỗi khi di chuyển {base}: {e}")

print(f"Đã di chuyển thành công {moved} file .wav sang {backup_d}")
