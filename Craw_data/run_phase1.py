"""
Phase 1: Sequential URL Collection — HTML Parser
Script chạy Phase 1 - Thu thập URL video từ các kênh YouTube.
Không dùng scrapetube, parse thẳng HTML / ytInitialData JSON.

Cách dùng:

    python Craw_data/run_phase1.py
"""

import sys
import os

# Thêm thư mục cha vào path để import module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from youtube_crawler.collector import URLCollector


def plot_channel_distribution(df):
    """Vẽ biểu đồ phân bố số URL theo kênh và lưu vào Youtube_Data/."""
    import matplotlib.pyplot as plt

    if df.empty or "channel_name" not in df.columns:
        print("Không có dữ liệu để vẽ biểu đồ.")
        return

    # Đếm số URL theo kênh
    channel_counts = df["channel_name"].value_counts()

    # Vẽ biểu đồ cột ngang
    fig, ax = plt.subplots(figsize=(10, max(4, len(channel_counts) * 0.5)))
    channel_counts.plot(kind="barh", ax=ax, color="steelblue")

    ax.set_xlabel("Số lượng URL")
    ax.set_ylabel("Kênh")
    ax.set_title("Phân bố số URL theo kênh YouTube")
    ax.invert_yaxis()  # Kênh nhiều nhất ở trên

    # Thêm số liệu trên mỗi thanh
    for i, (count, name) in enumerate(zip(channel_counts.values, channel_counts.index)):
        ax.text(count + 0.5, i, str(count), va="center", fontsize=9)

    plt.tight_layout()

    # Lưu biểu đồ
    output_path = "Craw_data/Youtube_Data/url_distribution.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Biểu đồ phân bố đã lưu: {output_path}")

    # In thống kê ra console
    print("\n--- Phân bố URL theo kênh ---")
    for name, count in channel_counts.items():
        print(f"  {name}: {count} URLs")


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("=== Phase 1: Sequential URL Collection (HTML Parser) ===")

    # -------------------------------------------------------------------------
    # Cấu hình thu thập — chỉnh theo nhu cầu
    # -------------------------------------------------------------------------
    MAX_URL_TOTAL       = 1   # Tổng URL tối đa cần thu thập
    MAX_URL_PER_CHANNEL = 1  # URL tối đa lấy từ mỗi kênh
    TIMEOUT_PER_CHANNEL = 120.0  # Giây timeout mỗi kênh
    REQUEST_TIMEOUT     = 30.0   # Giây timeout mỗi HTTP request

    collector = URLCollector(
        input_excel_path="Craw_data/Begin.xlsx",
        output_excel_path="Craw_data/Youtube_Data/video_urls.xlsx",
        max_url=MAX_URL_TOTAL,
        max_url_per_channel=MAX_URL_PER_CHANNEL,
        timeout_per_channel=TIMEOUT_PER_CHANNEL,
        request_timeout=REQUEST_TIMEOUT,
    )

    df = collector.collect_urls()

    print(f"\n=== Hoàn thành ===")
    print(f"Tổng số URLs thu thập : {len(df)}")
    print(f"max_url_per_channel   : {MAX_URL_PER_CHANNEL}")
    print(f"timeout_per_channel   : {TIMEOUT_PER_CHANNEL}s")

    # Vẽ biểu đồ phân bố
    plot_channel_distribution(df)


if __name__ == "__main__":
    main()
