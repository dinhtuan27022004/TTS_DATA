"""
Phase 1: URL Collection
Thu thập URL video từ các kênh YouTube bằng cách parse trực tiếp
HTML và ytInitialData JSON — không dùng scrapetube.

Cơ chế:
  1. GET https://www.youtube.com/@handle/videos  (hoặc /channel/UCxxx/videos)
  2. Trích xuất ytInitialData JSON từ thẻ <script>
  3. Parse video list từ richGridRenderer / gridRenderer
  4. Phân trang qua continuationToken → POST /youtubei/v1/browse
"""

import json
import os
import re
import time
import logging
from typing import List, Set, Optional, Tuple
from urllib.parse import unquote

import pandas as pd
import requests

from .models import VideoInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Hằng số YouTube internal API ──────────────────────────────────────────────
_YT_BROWSE_URL = "https://www.youtube.com/youtubei/v1/browse"
_YT_API_KEY    = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"  # public key (không đổi)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}

_BROWSE_PAYLOAD_TEMPLATE = {
    "context": {
        "client": {
            "clientName": "WEB",
            "clientVersion": "2.20240101.00.00",
            "hl": "vi",
        }
    }
}


class URLCollector:
    """
    Thu thập URL video từ các kênh YouTube bằng HTML parser.

    Duyệt lần lượt từng kênh:
      - Mỗi kênh lấy tối đa max_url_per_channel video
      - Hoặc dừng khi hết timeout_per_channel giây
      - Rồi chuyển sang kênh tiếp theo
    Dừng hoàn toàn khi đạt max_url tổng.
    """

    def __init__(
        self,
        input_excel_path: str = "Craw_data/Begin.xlsx",
        output_excel_path: str = "Youtube_Data/video_urls.xlsx",
        max_url: int = 1000,
        max_url_per_channel: int = 200,
        timeout_per_channel: float = 120.0,
        request_timeout: float = 30.0,
    ):
        """
        Args:
            input_excel_path: File Begin.xlsx chứa URL kênh (cột đầu tiên)
            output_excel_path: File Excel đầu ra
            max_url: Tổng số URL tối đa (điều kiện dừng toàn cục)
            max_url_per_channel: Số URL tối đa lấy từ mỗi kênh
            timeout_per_channel: Timeout (giây) cho mỗi kênh
            request_timeout: Timeout (giây) cho mỗi HTTP request
        """
        self.input_excel_path    = input_excel_path
        self.output_excel_path   = output_excel_path
        self.max_url             = max_url
        self.max_url_per_channel = max_url_per_channel
        self.timeout_per_channel = timeout_per_channel
        self.request_timeout     = request_timeout

        self.collected_urls: Set[str]    = set()
        self.results: List[VideoInfo]    = []
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def collect_urls(self) -> pd.DataFrame:
        """
        Thu thập video URLs từ tất cả kênh.

        Returns:
            DataFrame với columns: ['channel_name', 'title', 'url']
        """
        output_dir = os.path.dirname(self.output_excel_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Resume: nạp kết quả cũ
        self._load_existing_output()

        if len(self.collected_urls) >= self.max_url:
            logger.info(f"Đã đạt max_url ({self.max_url}). Không cần thu thập thêm.")
            return self._to_dataframe()

        # Đọc danh sách kênh
        channel_urls = self._read_channel_urls()
        logger.info(f"Đọc được {len(channel_urls)} kênh từ {self.input_excel_path}")

        for i, channel_url in enumerate(channel_urls, 1):
            if len(self.collected_urls) >= self.max_url:
                logger.info(f"Đạt max_url ({self.max_url}). Dừng thu thập.")
                break

            logger.info(f"[{i}/{len(channel_urls)}] Đang duyệt kênh: {channel_url}")
            count = self._collect_from_channel(channel_url)
            logger.info(
                f"  → Thu thập được {count} video từ kênh này. "
                f"Tổng: {len(self.collected_urls)}"
            )

            # Lưu sau mỗi kênh (để resume được)
            self._save_output(self._to_dataframe())

        df = self._to_dataframe()
        logger.info(f"Hoàn thành. Tổng cộng: {len(self.collected_urls)} URLs.")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Core: HTML parser + continuation
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_from_channel(self, channel_url: str) -> int:
        """
        Thu thập video từ 1 kênh bằng cách parse ytInitialData.

        Bước 1: GET trang /videos của kênh → trích ytInitialData
        Bước 2: Parse danh sách video từ richGridRenderer
        Bước 3: Nếu còn continuationToken → POST /youtubei/v1/browse để lấy thêm

        Returns:
            Số video thu thập được từ kênh này.
        """
        videos_url = self._build_videos_url(channel_url)
        if not videos_url:
            logger.warning(f"  Không parse được URL kênh: {channel_url}")
            return 0

        count = 0
        start_time = time.time()

        # ── Bước 1: Fetch trang /videos ────────────────────────────────────
        try:
            resp = self._session.get(videos_url, timeout=self.request_timeout)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"  Không thể tải trang kênh {videos_url}: {e}")
            return 0

        # ── Bước 2: Parse ytInitialData ────────────────────────────────────
        initial_data = self._extract_yt_initial_data(resp.text)
        if not initial_data:
            logger.error(f"  Không tìm thấy ytInitialData trong trang: {videos_url}")
            return 0

        # Lấy tên kênh từ metadata
        channel_name = self._get_channel_name(initial_data, channel_url)

        # Parse page 1
        videos, continuation_token = self._parse_videos_from_data(initial_data)

        for video_id, title in videos:
            if self._should_stop(count, start_time):
                return count
            if self._add_video(video_id, title, channel_name):
                count += 1

        # ── Bước 3: Pagination qua continuationToken ───────────────────────
        while continuation_token and not self._should_stop(count, start_time):
            time.sleep(0.5)  # lịch sự với server

            try:
                data = self._fetch_continuation(continuation_token)
            except Exception as e:
                logger.warning(f"  Lỗi continuation: {e}")
                break

            if not data:
                break

            videos, continuation_token = self._parse_videos_from_continuation(data)

            for video_id, title in videos:
                if self._should_stop(count, start_time):
                    return count
                if self._add_video(video_id, title, channel_name):
                    count += 1

        return count

    def _should_stop(self, count: int, start_time: float) -> bool:
        """Trả về True nếu cần dừng thu thập."""
        if len(self.collected_urls) >= self.max_url:
            return True
        if count >= self.max_url_per_channel:
            logger.info(f"  Đạt max_url_per_channel ({self.max_url_per_channel}). Chuyển kênh.")
            return True
        elapsed = time.time() - start_time
        if elapsed >= self.timeout_per_channel:
            logger.info(f"  Hết timeout ({self.timeout_per_channel}s). Chuyển kênh.")
            return True
        return False

    def _add_video(self, video_id: str, title: str, channel_name: str) -> bool:
        """Thêm video vào kết quả nếu chưa có. Trả về True nếu thêm thành công."""
        if not video_id:
            return False
        url = f"https://www.youtube.com/watch?v={video_id}"
        if url in self.collected_urls:
            return False
        self.results.append(VideoInfo(channel_name=channel_name, title=title, url=url))
        self.collected_urls.add(url)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # HTML / JSON parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_yt_initial_data(self, html: str) -> Optional[dict]:
        """
        Trích xuất ytInitialData JSON từ HTML trang YouTube.

        YouTube nhúng JSON này trong thẻ <script>:
            var ytInitialData = {...};
        """
        # Pattern chính
        match = re.search(
            r"var ytInitialData\s*=\s*(\{.*?\});\s*</script>",
            html, re.DOTALL
        )
        if not match:
            # Fallback: window["ytInitialData"]
            match = re.search(
                r'window\["ytInitialData"\]\s*=\s*(\{.*?\});\s*</script>',
                html, re.DOTALL
            )
        if not match:
            return None

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.warning(f"  Lỗi parse ytInitialData JSON: {e}")
            return None

    def _parse_videos_from_data(
        self, data: dict
    ) -> Tuple[List[Tuple[str, str]], Optional[str]]:
        """
        Parse danh sách video và continuationToken từ ytInitialData.

        Cấu trúc YouTube:
          data
          └─ contents
             └─ twoColumnBrowseResultsRenderer
                └─ tabs[]
                   └─ tabRenderer (active tab = Videos)
                      └─ content
                         └─ richGridRenderer / gridRenderer
                            └─ contents[]
                               ├─ richItemRenderer → videoRenderer
                               └─ continuationItemRenderer → continuationToken
        """
        videos: List[Tuple[str, str]] = []
        continuation_token: Optional[str] = None

        try:
            tabs = (
                data.get("contents", {})
                    .get("twoColumnBrowseResultsRenderer", {})
                    .get("tabs", [])
            )

            # Tìm tab Videos (tabRenderer với title "Videos" hoặc selected=True
            # và có content là richGridRenderer)
            grid_contents = []
            for tab in tabs:
                tab_renderer = tab.get("tabRenderer", {})
                content = tab_renderer.get("content", {})

                # richGridRenderer (layout mới)
                rich_grid = content.get("richGridRenderer", {})
                if rich_grid:
                    grid_contents = rich_grid.get("contents", [])
                    break

                # gridRenderer (layout cũ)
                section_list = content.get("sectionListRenderer", {})
                if section_list:
                    for section in section_list.get("contents", []):
                        item_section = section.get("itemSectionRenderer", {})
                        for item in item_section.get("contents", []):
                            grid = item.get("gridRenderer", {})
                            if grid:
                                grid_contents = grid.get("items", [])
                                break

            for item in grid_contents:
                # Layout mới: richItemRenderer
                rich_item = item.get("richItemRenderer", {})
                video_renderer = rich_item.get("content", {}).get("videoRenderer", {})

                # Layout cũ: gridVideoRenderer
                if not video_renderer:
                    video_renderer = item.get("gridVideoRenderer", {})

                if video_renderer:
                    vid, title = self._extract_video_info(video_renderer)
                    if vid:
                        videos.append((vid, title))
                    continue

                # Continuation token
                cont_item = item.get("continuationItemRenderer", {})
                if cont_item:
                    token = self._extract_continuation_token(cont_item)
                    if token:
                        continuation_token = token

        except Exception as e:
            logger.warning(f"  Lỗi parse video list: {e}")

        return videos, continuation_token

    def _parse_videos_from_continuation(
        self, data: dict
    ) -> Tuple[List[Tuple[str, str]], Optional[str]]:
        """
        Parse kết quả từ API continuation (/youtubei/v1/browse).

        Response structure:
          data
          └─ onResponseReceivedActions[]
             └─ appendContinuationItemsAction
                └─ continuationItems[]
                   ├─ richItemRenderer → videoRenderer
                   └─ continuationItemRenderer → continuationToken
        """
        videos: List[Tuple[str, str]] = []
        continuation_token: Optional[str] = None

        try:
            actions = data.get("onResponseReceivedActions", [])
            for action in actions:
                append_action = action.get("appendContinuationItemsAction", {})
                items = append_action.get("continuationItems", [])

                for item in items:
                    rich_item = item.get("richItemRenderer", {})
                    video_renderer = rich_item.get("content", {}).get("videoRenderer", {})

                    if not video_renderer:
                        video_renderer = item.get("gridVideoRenderer", {})

                    if video_renderer:
                        vid, title = self._extract_video_info(video_renderer)
                        if vid:
                            videos.append((vid, title))
                        continue

                    cont_item = item.get("continuationItemRenderer", {})
                    if cont_item:
                        token = self._extract_continuation_token(cont_item)
                        if token:
                            continuation_token = token

        except Exception as e:
            logger.warning(f"  Lỗi parse continuation response: {e}")

        return videos, continuation_token

    def _extract_video_info(self, video_renderer: dict) -> Tuple[str, str]:
        """Trích xuất (videoId, title) từ videoRenderer object."""
        video_id = video_renderer.get("videoId", "")

        # Title
        title_obj = video_renderer.get("title", {})
        if isinstance(title_obj, dict):
            runs = title_obj.get("runs", [])
            title = runs[0].get("text", "") if runs else title_obj.get("simpleText", "")
        else:
            title = str(title_obj)

        return video_id, title

    def _extract_continuation_token(self, cont_item: dict) -> Optional[str]:
        """Trích xuất continuationToken từ continuationItemRenderer."""
        try:
            return (
                cont_item
                .get("continuationEndpoint", {})
                .get("continuationCommand", {})
                .get("token", None)
            )
        except Exception:
            return None

    def _fetch_continuation(self, token: str) -> Optional[dict]:
        """
        POST đến /youtubei/v1/browse với continuationToken để lấy thêm video.
        """
        payload = {**_BROWSE_PAYLOAD_TEMPLATE, "continuation": token}
        try:
            resp = self._session.post(
                f"{_YT_BROWSE_URL}?key={_YT_API_KEY}",
                json=payload,
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"  Lỗi fetch continuation: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # URL building helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_videos_url(self, channel_url: str) -> Optional[str]:
        """
        Chuyển URL kênh thành URL trang /videos.

        Hỗ trợ:
          - https://www.youtube.com/@handle
          - https://www.youtube.com/channel/UCxxxx
          - https://www.youtube.com/c/name
        """
        url = unquote(channel_url.strip().rstrip("/"))

        # Đã có /videos → giữ nguyên
        if url.endswith("/videos"):
            return url

        # Các pattern hợp lệ
        for pattern in (
            r"youtube\.com/(@[\w.-]+)",
            r"youtube\.com/channel/(UC[\w-]+)",
            r"youtube\.com/c/([\w.-]+)",
            r"youtube\.com/user/([\w.-]+)",
        ):
            match = re.search(pattern, url)
            if match:
                slug = match.group(1)
                # @handle hoặc channel/UCxxx
                if slug.startswith("@") or slug.startswith("UC"):
                    base = f"https://www.youtube.com/{slug}"
                else:
                    base = f"https://www.youtube.com/c/{slug}"
                return f"{base}/videos"

        logger.warning(f"  Không nhận dạng được URL kênh: {channel_url}")
        return None

    def _get_channel_name(self, data: dict, fallback: str) -> str:
        """Lấy tên kênh từ ytInitialData metadata."""
        try:
            # Thử lấy từ header
            header = (
                data.get("header", {})
                    .get("c4TabbedHeaderRenderer", {})
            )
            name = header.get("title", "")
            if name:
                return name

            # Thử từ microformat
            microformat = (
                data.get("microformat", {})
                    .get("microformatDataRenderer", {})
            )
            return microformat.get("title", fallback)
        except Exception:
            return fallback

    # ─────────────────────────────────────────────────────────────────────────
    # I/O helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _read_channel_urls(self) -> List[str]:
        """Đọc file Begin.xlsx, lấy URL kênh từ cột đầu tiên."""
        df = pd.read_excel(self.input_excel_path, header=None)
        urls = df.iloc[:, 0].dropna().astype(str).tolist()
        return [u.strip() for u in urls if u.strip()]

    def _load_existing_output(self):
        """Resume: đọc output Excel hiện có."""
        if not os.path.exists(self.output_excel_path):
            logger.info("Không tìm thấy output Excel. Bắt đầu từ đầu.")
            return
        try:
            df = pd.read_excel(self.output_excel_path)
            if "url" in df.columns:
                for _, row in df.iterrows():
                    info = VideoInfo(
                        channel_name=str(row.get("channel_name", "")),
                        title=str(row.get("title", "")),
                        url=str(row["url"]),
                    )
                    self.results.append(info)
                    self.collected_urls.add(info.url)
                logger.info(
                    f"Resume: đã nạp {len(self.collected_urls)} URLs từ output hiện có."
                )
        except Exception as e:
            logger.warning(f"Không thể đọc output hiện có: {e}. Bắt đầu từ đầu.")

    def _to_dataframe(self) -> pd.DataFrame:
        """Chuyển danh sách VideoInfo thành DataFrame."""
        if not self.results:
            return pd.DataFrame(columns=["channel_name", "title", "url"])
        return pd.DataFrame(
            [{"channel_name": v.channel_name, "title": v.title, "url": v.url}
             for v in self.results]
        )

    def _save_output(self, df: pd.DataFrame):
        """Lưu DataFrame ra file Excel."""
        df.to_excel(self.output_excel_path, index=False)
