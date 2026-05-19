"""
Phase 2: Audio Download
Tải audio từ YouTube video dưới dạng WAV với tên UUID ngẫu nhiên.
Sử dụng yt-dlp để tải và convert sang WAV format.
"""

import os
import json
import uuid
import logging
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioDownloader:
    """
    Tải audio từ YouTube video dưới dạng WAV.
    
    - Đọc URLs từ output Excel của Phase 1
    - Tải audio bằng yt-dlp, convert sang WAV
    - Đặt tên UUID ngẫu nhiên cho mỗi file
    - Lưu mapping.json (url -> filename) để truy vết và resume
    - Lưu stats.json: total_files, total_duration_seconds, avg_duration_seconds
    - Resume: skip URLs đã có trong mapping.json
    """

    def __init__(
        self,
        urls_excel_path: str = "Youtube_Data/video_urls.xlsx",
        output_dir: str = "Youtube_Data/Step_0"
    ):
        """
        Args:
            urls_excel_path: File Excel chứa URLs (output của Phase 1)
            output_dir: Thư mục Step_0 để lưu file WAV
        """
        self.urls_excel_path = urls_excel_path
        self.output_dir = output_dir

        # Đường dẫn mapping.json và stats.json
        self.mapping_path = os.path.join(output_dir, "mapping.json")
        self.stats_path = os.path.join(output_dir, "stats.json")

        # Mapping url -> filename (dùng cho resume)
        self.mapping: Dict[str, str] = {}

    def download_all(self) -> Dict[str, str]:
        """
        Tải tất cả audio, resume từ vị trí dừng.
        
        - Đọc URLs từ Excel
        - Skip URLs đã tải (dựa trên mapping.json)
        - Lưu stats.json sau khi hoàn thành
        
        Returns:
            Dict mapping url -> filename.wav
        """
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)

        # Load mapping hiện có (resume)
        self._load_mapping()

        # Đọc danh sách URLs từ Excel
        urls = self._read_urls_from_excel()
        logger.info(f"Đọc được {len(urls)} URLs từ {self.urls_excel_path}")

        # Đếm số URLs cần tải (chưa có trong mapping)
        urls_to_download = [url for url in urls if url not in self.mapping]
        logger.info(f"Đã tải trước đó: {len(self.mapping)} files")
        logger.info(f"Cần tải thêm: {len(urls_to_download)} files")

        if not urls_to_download:
            logger.info("Tất cả URLs đã được tải. Không cần tải thêm.")
            self._save_stats()
            return self.mapping

        # Tải từng URL với progress bar
        for url in tqdm(urls_to_download, desc="Downloading audio"):
            filename = self._download_single(url)
            if filename:
                self.mapping[url] = filename
                # Lưu mapping sau mỗi file (để resume nếu bị gián đoạn)
                self._save_mapping()

        # Lưu stats cuối cùng
        self._save_stats()

        logger.info(f"Hoàn thành! Tổng cộng: {len(self.mapping)} files trong {self.output_dir}")
        return self.mapping

    def _read_urls_from_excel(self) -> list:
        """
        Đọc danh sách URLs từ output Excel của Phase 1.
        Lấy cột 'url' từ DataFrame.
        
        Returns:
            Danh sách URL strings
        """
        if not os.path.exists(self.urls_excel_path):
            logger.error(f"Không tìm thấy file Excel: {self.urls_excel_path}")
            return []

        try:
            df = pd.read_excel(self.urls_excel_path)
            if "url" not in df.columns:
                logger.error(f"File Excel không có cột 'url': {self.urls_excel_path}")
                return []

            urls = df["url"].dropna().astype(str).tolist()
            # Lọc bỏ URL rỗng
            urls = [url.strip() for url in urls if url.strip()]
            return urls
        except Exception as e:
            logger.error(f"Lỗi đọc file Excel: {e}")
            return []

    def _download_single(self, url: str) -> Optional[str]:
        """
        Tải 1 video, convert sang WAV, đặt tên UUID.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Tên file WAV (vd: "a1b2c3d4-e5f6-7890-abcd-ef1234567890.wav")
            hoặc None nếu lỗi
        """
        # Tạo tên file UUID
        filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(self.output_dir, filename)

        # Cấu hình yt-dlp
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path.replace(".wav", ".%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",  # Best quality
            }],
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }

        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Kiểm tra file output tồn tại
            if os.path.exists(output_path):
                logger.info(f"[OK] {url} -> {filename}")
                return filename
            else:
                logger.warning(f"[FAIL] File không tồn tại sau khi tải: {output_path}")
                return None

        except Exception as e:
            logger.warning(f"[FAIL] Không thể tải {url}: {e}")
            return None

    def _load_mapping(self):
        """
        Load mapping.json hiện có (cho resume).
        Nếu file chưa tồn tại thì bỏ qua.
        """
        if not os.path.exists(self.mapping_path):
            logger.info("Không tìm thấy mapping.json. Bắt đầu từ đầu.")
            return

        try:
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
            logger.info(f"Resume: đã nạp {len(self.mapping)} entries từ mapping.json")
        except Exception as e:
            logger.warning(f"Không thể đọc mapping.json: {e}. Bắt đầu từ đầu.")
            self.mapping = {}

    def _save_mapping(self):
        """Lưu mapping.json (url -> filename)."""
        try:
            with open(self.mapping_path, "w", encoding="utf-8") as f:
                json.dump(self.mapping, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi lưu mapping.json: {e}")

    def _save_stats(self):
        """
        Lưu stats.json: total_files, total_duration_seconds, avg_duration_seconds.
        Tính duration từ các file WAV trong output_dir.
        """
        import wave

        total_files = 0
        total_duration = 0.0

        # Duyệt tất cả file WAV trong output_dir
        for filename in os.listdir(self.output_dir):
            if not filename.endswith(".wav"):
                continue

            filepath = os.path.join(self.output_dir, filename)
            try:
                with wave.open(filepath, "r") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
                    total_duration += duration
                    total_files += 1
            except Exception as e:
                logger.warning(f"Không thể đọc duration của {filename}: {e}")
                # Vẫn đếm file nhưng không tính duration
                total_files += 1

        # Tính trung bình
        avg_duration = total_duration / total_files if total_files > 0 else 0.0

        stats = {
            "total_files": total_files,
            "total_duration_seconds": round(total_duration, 2),
            "avg_duration_seconds": round(avg_duration, 2)
        }

        try:
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"Stats: {total_files} files, {total_duration:.1f}s total, {avg_duration:.1f}s avg")
        except Exception as e:
            logger.error(f"Lỗi lưu stats.json: {e}")
