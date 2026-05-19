"""
Debug Phase 1: Kiểm tra tại sao không thu thập được URL.
- Đọc Begin.xlsx, hiển thị danh sách kênh
- Thử trích xuất channel_id / username từ mỗi URL
- Thử scrape từng kênh, ghi log chi tiết vào JSON
"""

import sys
import os
import json
import re
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

try:
    import scrapetube
    HAS_SCRAPETUBE = True
except ImportError:
    HAS_SCRAPETUBE = False

from youtube_crawler.collector import URLCollector


def extract_channel_id(url: str) -> str:
    match = re.search(r"/channel/(UC[\w-]+)", url)
    return match.group(1) if match else ""


def extract_channel_username(url: str) -> str:
    match = re.search(r"/@([\w.-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"/c/([\w.-]+)", url)
    if match:
        return match.group(1)
    return ""


def main():
    input_excel = "Craw_data/Begin.xlsx"
    output_json = "Craw_data/debug_phase1_result.json"

    report = {
        "input_file": input_excel,
        "scrapetube_installed": HAS_SCRAPETUBE,
        "channels": [],
        "summary": {}
    }

    # 1. Đọc Begin.xlsx
    print(f"Đọc file: {input_excel}")
    try:
        df = pd.read_excel(input_excel, header=None)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First 10 rows:\n{df.head(10)}")
    except Exception as e:
        report["error"] = f"Không đọc được Begin.xlsx: {e}"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Lỗi: {e}")
        return

    # 2. Lấy danh sách URL kênh
    urls = df.iloc[:, 0].dropna().astype(str).tolist()
    urls = [url.strip() for url in urls if url.strip()]
    print(f"\nTổng số URL kênh: {len(urls)}")

    total_found = 0
    total_skipped = 0
    total_errors = 0

    # 3. Duyệt từng kênh
    for i, channel_url in enumerate(urls):
        print(f"\n--- Kênh {i+1}/{len(urls)}: {channel_url} ---")

        channel_info = {
            "index": i + 1,
            "url": channel_url,
            "channel_id": extract_channel_id(channel_url),
            "channel_username": extract_channel_username(channel_url),
            "status": "unknown",
            "videos_found": [],
            "videos_skipped": [],
            "error": None,
            "total_found": 0,
            "total_skipped": 0
        }

        if not channel_info["channel_id"] and not channel_info["channel_username"]:
            channel_info["status"] = "rejected"
            channel_info["error"] = "Không trích xuất được channel_id hoặc username từ URL"
            print(f"  [REJECTED] Không parse được URL")
            total_skipped += 1
            report["channels"].append(channel_info)
            continue

        if not HAS_SCRAPETUBE:
            channel_info["status"] = "error"
            channel_info["error"] = "scrapetube chưa cài đặt"
            report["channels"].append(channel_info)
            continue

        # Thử scrape
        try:
            if channel_info["channel_id"]:
                print(f"  Scraping bằng channel_id: {channel_info['channel_id']}")
                videos_gen = scrapetube.get_channel(channel_id=channel_info["channel_id"])
            else:
                print(f"  Scraping bằng channel_url: {channel_url}")
                videos_gen = scrapetube.get_channel(channel_url=channel_url)

            count = 0
            max_check = 20  # Chỉ kiểm tra tối đa 20 video mỗi kênh để debug nhanh

            for video in videos_gen:
                if count >= max_check:
                    break

                video_id = video.get("videoId", "")
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                # Lấy title
                video_title = video.get("title", {})
                if isinstance(video_title, dict):
                    title_runs = video_title.get("runs", [])
                    title = title_runs[0].get("text", "") if title_runs else ""
                elif isinstance(video_title, list):
                    title = video_title[0].get("text", "") if video_title else ""
                else:
                    title = str(video_title)

                # Lấy channel name
                owner_text = video.get("ownerText", {})
                channel_name = ""
                if isinstance(owner_text, dict):
                    runs = owner_text.get("runs", [])
                    if runs:
                        channel_name = runs[0].get("text", "")

                # Kiểm tra video thuộc channel
                is_from_channel = True
                video_owner_id = ""
                if isinstance(owner_text, dict):
                    runs = owner_text.get("runs", [])
                    if runs:
                        nav_endpoint = runs[0].get("navigationEndpoint", {})
                        browse_endpoint = nav_endpoint.get("browseEndpoint", {})
                        video_owner_id = browse_endpoint.get("browseId", "")
                        if channel_info["channel_id"] and video_owner_id:
                            is_from_channel = (video_owner_id == channel_info["channel_id"])

                video_entry = {
                    "video_url": video_url,
                    "title": title,
                    "channel_name": channel_name,
                    "video_owner_id": video_owner_id,
                    "is_from_channel": is_from_channel
                }

                if is_from_channel:
                    channel_info["videos_found"].append(video_entry)
                    video_entry["reason"] = "accepted"
                else:
                    video_entry["reason"] = f"owner_id mismatch: video={video_owner_id}, expected={channel_info['channel_id']}"
                    channel_info["videos_skipped"].append(video_entry)

                count += 1

            channel_info["total_found"] = len(channel_info["videos_found"])
            channel_info["total_skipped"] = len(channel_info["videos_skipped"])

            if channel_info["total_found"] > 0:
                channel_info["status"] = "success"
            elif count == 0:
                channel_info["status"] = "empty"
                channel_info["error"] = "Kênh không trả về video nào (có thể kênh trống hoặc bị chặn)"
            else:
                channel_info["status"] = "all_skipped"
                channel_info["error"] = "Tất cả video bị skip do không match channel"

            total_found += channel_info["total_found"]
            total_skipped += channel_info["total_skipped"]

            print(f"  Found: {channel_info['total_found']}, Skipped: {channel_info['total_skipped']}")

        except Exception as e:
            channel_info["status"] = "error"
            channel_info["error"] = f"{type(e).__name__}: {str(e)}"
            print(f"  [ERROR] {e}")
            total_errors += 1

        report["channels"].append(channel_info)

    # Summary
    report["summary"] = {
        "total_channels": len(urls),
        "total_videos_found": total_found,
        "total_videos_skipped": total_skipped,
        "total_errors": total_errors,
        "channels_success": sum(1 for c in report["channels"] if c["status"] == "success"),
        "channels_empty": sum(1 for c in report["channels"] if c["status"] == "empty"),
        "channels_rejected": sum(1 for c in report["channels"] if c["status"] == "rejected"),
        "channels_error": sum(1 for c in report["channels"] if c["status"] == "error"),
        "channels_all_skipped": sum(1 for c in report["channels"] if c["status"] == "all_skipped"),
    }

    # Lưu JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"KẾT QUẢ DEBUG:")
    print(f"  Tổng kênh: {report['summary']['total_channels']}")
    print(f"  Kênh thành công: {report['summary']['channels_success']}")
    print(f"  Kênh trống: {report['summary']['channels_empty']}")
    print(f"  Kênh bị reject (URL sai): {report['summary']['channels_rejected']}")
    print(f"  Kênh lỗi: {report['summary']['channels_error']}")
    print(f"  Tổng video tìm thấy: {report['summary']['total_videos_found']}")
    print(f"  Tổng video bị skip: {report['summary']['total_videos_skipped']}")
    print(f"\nChi tiết đã lưu vào: {output_json}")


if __name__ == "__main__":
    main()
