"""
Phase 1: URL Collection
Thu thập URL video từ các kênh YouTube.

Duyệt lần lượt từng kênh, mỗi kênh lấy tối đa max_per_channel video
hoặc dừng khi hết timeout cho kênh đó, rồi chuyển sang kênh tiếp theo.
"""

import os
import re
import time
import logging
from typing import List, Set, Optional

import pandas as pd
import scrapetube

from .models import VideoInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class URLCollector:
    """
    Thu thập URL video từ các kênh YouTube.

    Duyệt lần lượt từng kênh:
      - Mỗi kênh lấy tối đa max_per_channel video
      - Hoặc dừng khi hết channel_timeout giây
      - Rồi chuyển sang kênh tiếp theo
    Dừng hoàn toàn khi đạt max_url tổng.
    """

    def __init__(
        self,
        input_excel_path: str = "Craw_data/Begin.xlsx",
        output_excel_path: str = "Youtube_Data/video_urls.xlsx",
        max_url: int = 1000,
        max_per_channel: int = 200,
        channel_timeout: float = 120.0,
    ):
        """
        Args:
            input_excel_path: File Begin.xlsx chứa URL kênh (cột đầu tiên)
            output_excel_path: File Excel đầu ra
            max_url: Tổng số URL tối đa (điều kiện dừng toàn cục)
            max_per_channel: Số URL tối đa lấy từ mỗi kênh
            channel_timeout: Timeout (giây) cho mỗi kênh, hết thời gian thì chuyển kênh
        """
        self.input_excel_path = input_excel_path
        self.output_excel_path = output_excel_path
        self.max_url = max_url
        self.max_per_channel = max_per_channel
        self.channel_timeout = channel_timeout

        self.collected_urls: Set[str] = set()
        self.results: List[VideoInfo] = []

    def collect_urls(self) -> pd.DataFrame:
        """
        Thu thập video URLs từ tất cả kênh.

        Duyệt lần lượt từng kênh, mỗi kênh lấy tối đa max_per_channel
        hoặc dừng khi hết channel_timeout. Dừng hoàn toàn khi đạt max_url.

        Returns:
            DataFrame với columns: ['channel_name', 'title', 'url']
        """
        output_dir = os.path.dirname(self.output_excel_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Resume
        self._load_existing_output()

        if len(self.collected_urls) >= self.max_url:
            logger.info(f"Đã đạt max_url ({self.max_url}). Không cần thu thập thêm.")
            return self._to_dataframe()

        # Đọc danh sách kênh
        channel_urls = self._read_channel_urls()
        logger.info(f"Đọc được {len(channel_urls)} kênh từ {self.input_excel_path}")

        # Duyệt lần lượt từng kênh
        for i, channel_url in enumerate(channel_urls, 1):
            if len(self.collected_urls) >= self.max_url:
                logger.info(f"Đạt max_url ({self.max_url}). Dừng thu thập.")
                break

            logger.info(f"[{i}/{len(channel_urls)}] Đang duyệt kênh: {channel_url}")
            count = self._collect_from_channel(channel_url)
            logger.info(f"  → Thu thập được {count} video từ kênh này. Tổng: {len(self.collected_urls)}")

            # Lưu sau mỗi kênh (để resume được)
            df = self._to_dataframe()
            self._save_output(df)

        df = self._to_dataframe()
        logger.info(f"Hoàn thành. Tổng cộng: {len(self.collected_urls)} URLs.")
        return df

    def _collect_from_channel(self, channel_url: str) -> int:
        """
        Thu thập video từ 1 kênh.
        Dừng khi: đạt max_per_channel, hết timeout, hoặc đạt max_url tổng.

        Returns:
            Số video thu thập được từ kênh này.
        """
        channel_id = self._extract_channel_id(channel_url)
        channel_username = self._extract_channel_username(channel_url)

        if not channel_id and not channel_username:
            logger.warning(f"  Không parse được URL: {channel_url}")
            return 0

        # Tạo generator
        try:
            if channel_id:
                gen = scrapetube.get_channel(channel_id=channel_id)
            else:
                gen = scrapetube.get_channel(channel_username=channel_username)
        except Exception as e:
            logger.error(f"  Không thể tạo generator: {e}")
            return 0

        count = 0
        start_time = time.time()

        for video in gen:
            # Điều kiện dừng: max_url tổng
            if len(self.collected_urls) >= self.max_url:
                break

            # Điều kiện dừng: max_per_channel
            if count >= self.max_per_channel:
                logger.info(f"  Đạt max_per_channel ({self.max_per_channel}). Chuyển kênh.")
                break

            # Điều kiện dừng: timeout
            elapsed = time.time() - start_time
            if elapsed >= self.channel_timeout:
                logger.info(f"  Hết timeout ({self.channel_timeout}s). Chuyển kênh.")
                break

            # Xử lý video
            video_id = video.get("videoId", "")
            if not video_id:
                continue

            video_url = f"https://www.youtube.com/watch?v={video_id}"

            # Trùng lặp
            if video_url in self.collected_urls:
                logger.debug(f"  [SKIP] Duplicate: {video_url}")
                continue

            # Verify channel ownership
            if channel_id and not self._is_video_from_channel(video, channel_id):
                logger.debug(f"  [SKIP] Not from channel: {video_url}")
                continue

            # Lấy metadata
            title = self._extract_title(video)
            channel_name = self._get_channel_name_from_video(video, channel_url)

            # Thêm vào kết quả
            video_info = VideoInfo(
                channel_name=channel_name,
                title=title,
                url=video_url
            )
            self.results.append(video_info)
            self.collected_urls.add(video_url)
            count += 1

            if count % 10 == 0:
                logger.info(f"  ... {count} video (elapsed: {time.time() - start_time:.1f}s)")

        return count

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _read_channel_urls(self) -> List[str]:
        """Đọc file Begin.xlsx, lấy URL kênh từ cột đầu tiên."""
        df = pd.read_excel(self.input_excel_path, header=None)
        urls = df.iloc[:, 0].dropna().astype(str).tolist()
        urls = [url.strip() for url in urls if url.strip()]
        return urls

    def _load_existing_output(self):
        """Resume: đọc output Excel hiện có."""
        if not os.path.exists(self.output_excel_path):
            logger.info("Không tìm thấy output Excel hiện có. Bắt đầu từ đầu.")
            return

        try:
            df = pd.read_excel(self.output_excel_path)
            if "url" in df.columns:
                for _, row in df.iterrows():
                    video_info = VideoInfo(
                        channel_name=str(row.get("channel_name", "")),
                        title=str(row.get("title", "")),
                        url=str(row["url"])
                    )
                    self.results.append(video_info)
                    self.collected_urls.add(video_info.url)
                logger.info(f"Resume: đã nạp {len(self.collected_urls)} URLs từ output hiện có.")
        except Exception as e:
            logger.warning(f"Không thể đọc output hiện có: {e}. Bắt đầu từ đầu.")

    def _is_video_from_channel(self, video: dict, channel_id: str) -> bool:
        """Kiểm tra video có thuộc channel_id không."""
        owner_text = video.get("ownerText", {})
        if isinstance(owner_text, dict):
            runs = owner_text.get("runs", [])
            if runs:
                nav_endpoint = runs[0].get("navigationEndpoint", {})
                browse_endpoint = nav_endpoint.get("browseEndpoint", {})
                video_owner_id = browse_endpoint.get("browseId", "")
                if video_owner_id:
                    return video_owner_id == channel_id
        # Mặc định cho qua (scrapetube đã lọc theo kênh)
        return True

    def _extract_title(self, video: dict) -> str:
        """Trích xuất title từ video metadata."""
        video_title = video.get("title", {})
        if isinstance(video_title, dict):
            title_runs = video_title.get("runs", [])
            return title_runs[0].get("text", "") if title_runs else ""
        elif isinstance(video_title, list):
            return video_title[0].get("text", "") if video_title else ""
        return str(video_title)

    def _get_channel_name_from_video(self, video: dict, channel_url: str) -> str:
        """Lấy tên kênh từ metadata video."""
        for key in ("ownerText", "shortBylineText"):
            field = video.get(key, {})
            if isinstance(field, dict):
                runs = field.get("runs", [])
                if runs:
                    return runs[0].get("text", "")
        return channel_url

    def _extract_channel_id(self, url: str) -> str:
        """Trích xuất channel ID từ URL dạng /channel/UC..."""
        match = re.search(r"/channel/(UC[\w-]+)", url)
        return match.group(1) if match else ""

    def _extract_channel_username(self, url: str) -> str:
        """Trích xuất username từ URL dạng /@username hoặc /c/username."""
        # Decode URL-encoded characters first
        from urllib.parse import unquote
        url = unquote(url)

        match = re.search(r"/@([\w.-]+)", url)
        if match:
            return match.group(1)
        match = re.search(r"/c/([\w.-]+)", url)
        if match:
            return match.group(1)
        return ""

    def _to_dataframe(self) -> pd.DataFrame:
        """Chuyển danh sách VideoInfo thành DataFrame."""
        if not self.results:
            return pd.DataFrame(columns=["channel_name", "title", "url"])
        data = [{"channel_name": v.channel_name, "title": v.title, "url": v.url}
                for v in self.results]
        return pd.DataFrame(data)

    def _save_output(self, df: pd.DataFrame):
        """Lưu DataFrame ra file Excel."""
        df.to_excel(self.output_excel_path, index=False)
