"""
Phase 3 – Visualization.

Vẽ đồ thị so sánh các checkpoint bằng seaborn + matplotlib:
  1. Bar chart:  giá trị mean mỗi metric mỗi checkpoint
  2. Box plot:   phân phối điểm (distribution) mỗi metric
  3. Heatmap:    bảng tổng hợp mean tất cả metric × checkpoint
"""
import json
import logging
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Metric display config: (key_in_json, label, lower_is_better)
METRIC_CONFIG = [
    ("pesq",        "PESQ",       False),
    ("stoi",        "STOI",       False),
    ("utmos",       "UTMOS",      False),
    ("f0_corr",     "F0 Corr",    False),
    ("speaker_sim", "Spk Sim",    False),
    ("wer",         "WER",        True),
    ("cer",         "CER",        True),
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_all_records(results_dir: str, dataset_name: str, metric_keys: Optional[List[str]] = None) -> list:
    """Đọc toàn bộ metric JSON, trả về list records dạng flat."""
    records = []
    dataset_dir = os.path.join(results_dir, dataset_name)
    if not os.path.isdir(dataset_dir):
        logger.warning("Không tìm thấy thư mục kết quả: %s", dataset_dir)
        return records

    valid_metrics = set(metric_keys) if metric_keys is not None else {m for m, _, _ in METRIC_CONFIG}

    for ckpt_name in sorted(os.listdir(dataset_dir)):
        ckpt_dir = os.path.join(dataset_dir, ckpt_name)
        if not os.path.isdir(ckpt_dir):
            continue
        for fname in os.listdir(ckpt_dir):
            if not fname.endswith(".json"):
                continue
            metric = fname[:-5]  # strip .json
            if metric not in valid_metrics:
                continue
            path = os.path.join(ckpt_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for s in data.get("samples", []):
                    if s.get("value") is not None:
                        records.append({
                            "checkpoint": ckpt_name,
                            "metric":     metric,
                            "value":      float(s["value"]),
                        })
            except Exception as exc:
                logger.warning("Không đọc được %s: %s", path, exc)

    return records


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _make_grid(n: int, max_cols: int = 3):
    """Tạo fig/axes lưới phù hợp với n ô."""
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    return fig, axes.flatten()


# ─── Phase 3 entry point ──────────────────────────────────────────────────────

def plot_comparison(
    results_dir: str,
    dataset_name: str,
    output_dir: str,
    metric_keys: Optional[List[str]] = None,
    filename_suffix: str = "",
    title_suffix: str = "",
) -> List[str]:
    """Vẽ các đồ thị so sánh checkpoint và lưu ra output_dir.

    Args:
        results_dir:  evaluate/results/
        dataset_name: Tên dataset.
        output_dir:   Thư mục lưu file ảnh chart.

    Returns:
        Danh sách đường dẫn các file ảnh đã lưu.
    """
    try:
        import seaborn as sns
        import pandas as pd
    except ImportError:
        logger.error("Thiếu seaborn/pandas. Cài: pip install seaborn pandas")
        return []

    records = _load_all_records(results_dir, dataset_name, metric_keys=metric_keys)
    if not records:
        logger.warning("Không có dữ liệu metric để vẽ đồ thị.")
        return []

    df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="husl", font_scale=1.0)

    # Metrics có đủ dữ liệu
    available = df["metric"].unique()
    active_cfg = [(m, lbl, lb) for m, lbl, lb in METRIC_CONFIG if m in available and (metric_keys is None or m in metric_keys)]

    if not active_cfg:
        logger.warning("Không có metric nào có dữ liệu.")
        return []

    saved: List[str] = []
    mean_df = df.groupby(["checkpoint", "metric"])["value"].mean().reset_index()

    # ── 1. Bar chart: mean per metric per checkpoint ──────────────────────────
    n = len(active_cfg)
    fig, axes_flat = _make_grid(n)

    for i, (metric, label, lower_better) in enumerate(active_cfg):
        ax  = axes_flat[i]
        dat = mean_df[mean_df["metric"] == metric].copy()
        sns.barplot(data=dat, x="checkpoint", y="value", hue="checkpoint", legend=False, ax=ax, palette="husl")
        direction = "↓ thấp hơn tốt hơn" if lower_better else "↑ cao hơn tốt hơn"
        ax.set_title(f"{label}  ({direction})", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)

        # Annotate bar value
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.002,
                    f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"So sánh checkpoint – {dataset_name}{title_suffix}", fontsize=13, y=1.01)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f"{dataset_name}{filename_suffix}_bar_chart.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(bar_path)
    logger.info("Bar chart lưu tại: %s", bar_path)

    # ── 2. Box plot: distribution per metric ─────────────────────────────────
    fig2, axes2_flat = _make_grid(n)

    for i, (metric, label, _) in enumerate(active_cfg):
        ax  = axes2_flat[i]
        dat = df[df["metric"] == metric]
        sns.boxplot(data=dat, x="checkpoint", y="value", hue="checkpoint", legend=False, ax=ax, palette="husl")
        ax.set_title(f"{label} – Phân phối", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)

    for j in range(i + 1, len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    fig2.suptitle(f"Phân phối metric – {dataset_name}{title_suffix}", fontsize=13, y=1.01)
    plt.tight_layout()
    box_path = os.path.join(output_dir, f"{dataset_name}{filename_suffix}_box_plot.png")
    fig2.savefig(box_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    saved.append(box_path)
    logger.info("Box plot lưu tại: %s", box_path)

    # ── 3. Heatmap: mean per checkpoint × metric ─────────────────────────────
    ordered_metrics = [m for m, _, _ in METRIC_CONFIG if m in available and (metric_keys is None or m in metric_keys)]
    pivot = mean_df.pivot(index="checkpoint", columns="metric", values="value")
    pivot = pivot.reindex(columns=ordered_metrics)

    n_ckpt   = len(pivot)
    n_metric = len(ordered_metrics)
    figw = max(8, n_metric * 1.8)
    figh = max(4, n_ckpt * 1.4)

    fig3, ax3 = plt.subplots(figsize=(figw, figh))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        ax=ax3, linewidths=0.5, cbar_kws={"shrink": 0.7},
    )
    ax3.set_title(f"Summary – {dataset_name}{title_suffix}", fontsize=13)
    ax3.set_xlabel("Metric")
    ax3.set_ylabel("Checkpoint")
    plt.tight_layout()
    heat_path = os.path.join(output_dir, f"{dataset_name}{filename_suffix}_heatmap.png")
    fig3.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    saved.append(heat_path)
    logger.info("Heatmap lưu tại: %s", heat_path)

    return saved
