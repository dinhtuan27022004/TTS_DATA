import os

source_dir = '/home/reg/TTS_DATA/Processed_DATA/PhoAudioBook'
part1_dir = '/home/reg/TTS_DATA/Processed_DATA/PhoAudioBook_part1'
part2_dir = '/home/reg/TTS_DATA/Processed_DATA/PhoAudioBook_part2'

os.makedirs(part1_dir, exist_ok=True)
os.makedirs(part2_dir, exist_ok=True)

print("Đang quét thư mục...")
wav_files = []
try:
    for entry in os.scandir(source_dir):
        if entry.is_file() and entry.name.endswith('.wav'):
            wav_files.append(entry.name)
except Exception as e:
    print("Error scanning:", e)

print(f"Tổng cộng {len(wav_files)} file wav.")

# Tính toán số lượng cho 200GB (tổng là 431GB -> tỷ lệ 200/431 = ~46.4%)
TARGET_COUNT = 450891

print(f"Sẽ chuyển {TARGET_COUNT} cặp file sang part 1 (tương đương khoảng 200GB)...")

for i, wav_name in enumerate(wav_files):
    base_name = wav_name[:-4]
    txt_name = base_name + '.txt'
    
    src_wav = os.path.join(source_dir, wav_name)
    src_txt = os.path.join(source_dir, txt_name)
    
    dest_dir = part1_dir if i < TARGET_COUNT else part2_dir
        
    try:
        if os.path.exists(src_wav):
            os.rename(src_wav, os.path.join(dest_dir, wav_name))
        if os.path.exists(src_txt):
            os.rename(src_txt, os.path.join(dest_dir, txt_name))
    except Exception as e:
        pass

    if (i+1) % 100000 == 0:
        print(f"Đã xử lý {i+1} cặp...")

print("Hoàn tất chia đôi dữ liệu! Bạn có thể bắt đầu quá trình zip PhoAudioBook_part1.")
