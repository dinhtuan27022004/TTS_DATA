#!/bin/bash
set -e

echo "============================================="
echo "   BẮT ĐẦU QUÁ TRÌNH ĐÓNG GÓI VÀ UPLOAD      "
echo "============================================="

# 1. Kiểm tra và xóa file .txt
echo ""
echo "[1/3] Đang kiểm tra và dọn dẹp các file .txt..."
TXT_COUNT=$(find Craw_data/Youtube_Data/Step_2 -name "*.txt" | wc -l)
if [ "$TXT_COUNT" -gt 0 ]; then
    find Craw_data/Youtube_Data/Step_2 -name "*.txt" -type f -delete
    echo " -> Đã xóa sạch $TXT_COUNT file .txt."
else
    echo " -> Hoàn hảo! Không có file .txt rác nào."
fi

# 2. Thống kê file .wav
echo ""
echo "[2/3] Đang thống kê danh sách file .wav..."
WAV_COUNT=$(find Craw_data/Youtube_Data/Step_2 -name "*.wav" | wc -l)
LAST_FILE=$(find Craw_data/Youtube_Data/Step_2 -name "*.wav" | sort | tail -n 1 | xargs basename)
echo " -> Tổng cộng có: $WAV_COUNT file .wav."
echo " -> File âm thanh cuối cùng (theo ABC) là: $LAST_FILE"

# 3. Nén và Upload thẳng lên mây (Stream)
echo ""
echo "[3/3] Đang nén (zip) và upload trực tiếp lên Google Drive..."
echo " -> Rclone sẽ báo cáo tốc độ (MB/s) và tiến trình liên tục ngay bên dưới đây:"
echo "---------------------------------------------"

cd Craw_data/Youtube_Data/Step_2

# -q: nén im lặng để không in 12,000 dòng log gây rối màn hình
# -r -: nén thành luồng dữ liệu (stdout)
# rclone rcat: hứng luồng dữ liệu và up lên mây
# -P: hiện thanh tiến trình cho rclone
zip -q -r - . -i "*.wav" | rclone rcat KhoiDriver:wav_data.zip --drive-root-folder-id 1RkVYTF0sUkSVSWQAna1l5-8hDO3put6x -P

cd ../../../

echo "---------------------------------------------"
echo "🎉 HOÀN TẤT TUYỆT ĐỐI! File wav_data.zip đã nằm gọn trên Google Drive."
echo "============================================="
