#!/bin/bash
set -e

# 1. Chạy script chia thư mục (nó dùng lệnh di chuyển file nên file gốc ở PhoAudioBook sẽ tự động biến mất)
echo "1. Đang chia thư mục PhoAudioBook..."
python3 /home/reg/TTS_DATA/split_pho.py

# Sau khi chia xong, thư mục PhoAudioBook sẽ trống không, ta xóa nó đi cho gọn
rm -rf /home/reg/TTS_DATA/Processed_DATA/PhoAudioBook

# 2. Nén phần 1
echo "2. Đang nén PhoAudioBook_part1..."
cd /home/reg/TTS_DATA/Processed_DATA
zip -r PhoAudioBook_part1.zip PhoAudioBook_part1

# 3. Upload phần 1 lên Google Drive
echo "3. Đang upload PhoAudioBook_part1.zip..."
rclone copy PhoAudioBook_part1.zip gdrive: --drive-root-folder-id 122YRQRkr1dONP-rzxkSUsvXgkeQVBisB

# 4. Xóa file zip phần 1 để giải phóng dung lượng
echo "4. Xóa file zip phần 1..."
rm -f PhoAudioBook_part1.zip

# 5. Nén phần 2
echo "5. Đang nén PhoAudioBook_part2..."
zip -r PhoAudioBook_part2.zip PhoAudioBook_part2

# 6. Upload phần 2
echo "6. Đang upload PhoAudioBook_part2.zip..."
rclone copy PhoAudioBook_part2.zip gdrive: --drive-root-folder-id 122YRQRkr1dONP-rzxkSUsvXgkeQVBisB

# 7. Xóa file zip phần 2
echo "7. Xóa file zip phần 2..."
rm -f PhoAudioBook_part2.zip

echo "HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!"
