import os

# 1. Đường dẫn tuyệt đối đến thư mục CHA chứa các thư mục con dataset
# Ví dụ: "/home/reg/TTS_DATA/Processed_DATA" chứa "PhoAudioBook", "libritts", ...
SOURCE_PARENT_DIR = "/home/reg/TTS_DATA/Processed_DATA"

# 2. Định nghĩa thư mục đích (DEST_DIR) - Sử dụng đường dẫn tương đối
# Để đảm bảo khi di chuyển dự án (folder F5-TTS-Vietnamese) đi nơi khác vẫn hoạt động đúng,
# ta sẽ xác định đường dẫn tuyệt đối của thư mục đích dựa trên vị trí của chính script này.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_DIR = os.path.join(SCRIPT_DIR, "data", "your_dataset")

def create_symlinks():
    # Kiểm tra sự tồn tại của thư mục cha nguồn
    if not os.path.exists(SOURCE_PARENT_DIR):
        print(f"Lỗi: Thư mục cha nguồn không tồn tại: {SOURCE_PARENT_DIR}")
        return

    # Tự động quét tất cả các thư mục con cấp 1 bên trong thư mục cha
    source_dirs = []
    for name in os.listdir(SOURCE_PARENT_DIR):
        full_path = os.path.join(SOURCE_PARENT_DIR, name)
        if os.path.isdir(full_path):
            source_dirs.append(full_path)

    if not source_dirs:
        print(f"Không tìm thấy thư mục con nào bên trong: {SOURCE_PARENT_DIR}")
        return

    print(f"Tìm thấy các thư mục dataset con: {[os.path.basename(d) for d in source_dirs]}")

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Thư mục đích: {DEST_DIR}")
    
    count = 0
    for src_dir in source_dirs:
        print(f"Đang tạo symlink từ: {src_dir} ...")
        # Lấy tên của thư mục con nguồn để làm prefix chống trùng tên file
        src_parent_name = os.path.basename(os.path.normpath(src_dir))
        
        # Duyệt đệ quy tất cả các file trong thư mục nguồn này
        for root, _, files in os.walk(src_dir):
            for file in files:
                # Chỉ lọc các file .wav và .txt
                if file.endswith(('.wav', '.txt')):
                    src_file_path = os.path.abspath(os.path.join(root, file))
                    
                    # Tạo tên file mới để tránh trùng lặp: [tên_thư_mục_nguồn]_[đường_dẫn_tương_đối].wav/txt
                    rel_path = os.path.relpath(src_file_path, src_dir)
                    new_filename = f"{src_parent_name}_{rel_path.replace(os.sep, '_')}"
                    
                    dest_file_path = os.path.join(DEST_DIR, new_filename)
                    
                    # Nếu link đã tồn tại hoặc bị hỏng, xóa đi để tạo lại link mới nhất
                    if os.path.exists(dest_file_path) or os.path.islink(dest_file_path):
                        os.unlink(dest_file_path)
                    
                    # Tạo symlink sử dụng đường dẫn tuyệt đối của file nguồn.
                    # Khi di chuyển thư mục dự án F5-TTS-Vietnamese đi nơi khác,
                    # các symlink này vẫn hoạt động tốt vì nguồn trỏ tới là đường dẫn tuyệt đối cố định.
                    os.symlink(src_file_path, dest_file_path)
                    count += 1

    print(f"Hoàn thành! Đã tạo thành công {count} symlink tại {DEST_DIR}.")

if __name__ == "__main__":
    create_symlinks()
