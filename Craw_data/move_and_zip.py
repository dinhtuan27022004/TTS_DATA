import os
import shutil
import glob
from tqdm import tqdm
import subprocess

src_dir = "Youtube_Data/Step_2"
dest_dir = "Youtube_Data/YT_DT_P1"

print("Đang tạo thư mục đích...")
os.makedirs(dest_dir, exist_ok=True)

print("Đang tìm các file .txt...")
txt_files = glob.glob(os.path.join(src_dir, "*.txt"))
txt_files.sort()
txt_files = txt_files[:10000]

print(f"Đã chọn {len(txt_files)} file txt để di chuyển.")

for txt_path in tqdm(txt_files, desc="Đang di chuyển"):
    basename = os.path.basename(txt_path)
    wav_basename = basename.replace(".txt", ".wav")
    wav_path = os.path.join(src_dir, wav_basename)
    
    # Move txt
    shutil.move(txt_path, os.path.join(dest_dir, basename))
    
    # Move wav if exists
    if os.path.exists(wav_path):
        shutil.move(wav_path, os.path.join(dest_dir, wav_basename))

print("Hoàn thành di chuyển 10000 cặp file. Đang nén file ZIP...")
subprocess.run(["zip", "-r", "-q", "YT_DT_P1.zip", dest_dir])
print("Hoàn thành nén file YT_DT_P1.zip!")
