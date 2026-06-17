import pandas as pd
import json
import os

def main():
    excel_path = "Youtube_Data/video_urls.xlsx"
    mapping_path = "Youtube_Data/Step_0/mapping.json"

    # Kiểm tra xem file Excel có tồn tại không
    if not os.path.exists(excel_path):
        print(f"Lỗi: Không tìm thấy file {excel_path}")
        return

    # Đảm bảo thư mục đích tồn tại
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

    print(f"Đang đọc danh sách từ {excel_path}...")
    df = pd.read_excel(excel_path)
    
    if "url" not in df.columns:
        print("Lỗi: Không tìm thấy cột 'url' trong file Excel!")
        return

    urls = df["url"].dropna().astype(str).tolist()

    # Tạo mapping đánh dấu tất cả các URL này đã được xử lý
    mapping = {}
    for url in urls:
        url = url.strip()
        if url:
            mapping[url] = "already_downloaded_elsewhere.wav"

    # Ghi ra JSON
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Thành công! Đã tạo file {mapping_path} chứa {len(mapping)} URLs.")
    print("Các script crawler từ nay sẽ TỰ ĐỘNG BỎ QUA các URL này!")

if __name__ == "__main__":
    main()
