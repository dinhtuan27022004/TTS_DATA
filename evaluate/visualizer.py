"""
Visualizer module cho hệ thống đánh giá TTS.

Tạo các biểu đồ trực quan hóa kết quả đánh giá:
- Bar chart: giá trị trung bình từng metric
- Distribution plot: phân phối điểm từng metric
- Radar chart: tổng quan các metric (normalized)
- Comparison plot: so sánh nhiều model
"""

import json
import logging
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate.models import MetricResult

logger = logging.getLogger(__name__)

# Danh sách metric names tương ứng với attributes trong MetricResult
METRIC_NAMES = ["mcd", "pesq", "stoi", "utmos", "f0_correlation", "wer"]


class Visualizer:
    """Tạo biểu đồ trực quan hóa kết quả đánh giá TTS.

    Hỗ trợ tạo bar chart, distribution plot, radar chart, và comparison plot.
    Tất cả biểu đồ được lưu dưới dạng file PNG.
    """

    def _extract_metric_values(
        self, results: List[MetricResult]
    ) -> Dict[str, List[float]]:
        """Trích xuất giá trị metric từ danh sách MetricResult.

        Args:
            results: Danh sách MetricResult.

        Returns:
            Dict mapping metric name -> list of non-None values.
        """
        metric_values: Dict[str, List[float]] = {name: [] for name in METRIC_NAMES}
        for result in results:
            for name in METRIC_NAMES:
                value = getattr(result, name, None)
                if value is not None:
                    metric_values[name].append(value)
        return metric_values

    def plot_bar_chart(self, results: List[MetricResult], output_dir: str) -> str:
        """Tạo bar chart hiển thị giá trị trung bình của từng metric.

        Args:
            results: Danh sách MetricResult chứa kết quả đánh giá.
            output_dir: Thư mục lưu biểu đồ.

        Returns:
            Đường dẫn file PNG đã lưu.
        """
        os.makedirs(output_dir, exist_ok=True)

        metric_values = self._extract_metric_values(results)

        # Chỉ plot metric có dữ liệu
        names = []
        means = []
        for name in METRIC_NAMES:
            if metric_values[name]:
                names.append(name.upper())
                means.append(float(np.mean(metric_values[name])))

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, means, color="steelblue", edgecolor="black")

        # Thêm giá trị trên mỗi bar
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{mean:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Metric")
        ax.set_ylabel("Mean Value")
        ax.set_title("TTS Evaluation - Mean Metric Values")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "bar_chart.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Bar chart saved to: {output_path}")
        return output_path

    def plot_distribution(self, results: List[MetricResult], output_dir: str) -> str:
        """Tạo distribution plot hiển thị phân phối điểm của từng metric.

        Args:
            results: Danh sách MetricResult chứa kết quả đánh giá.
            output_dir: Thư mục lưu biểu đồ.

        Returns:
            Đường dẫn file PNG đã lưu.
        """
        os.makedirs(output_dir, exist_ok=True)

        metric_values = self._extract_metric_values(results)

        # Đếm số metric có dữ liệu
        active_metrics = [
            (name, values)
            for name, values in metric_values.items()
            if values
        ]

        if not active_metrics:
            logger.warning("No metric data available for distribution plot.")
            # Tạo empty plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("TTS Evaluation - Score Distribution (No Data)")
            output_path = os.path.join(output_dir, "distribution.png")
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        n_metrics = len(active_metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        for idx, (name, values) in enumerate(active_metrics):
            ax = axes[idx]
            ax.hist(values, bins=min(20, max(5, len(values) // 2)), 
                    color="steelblue", edgecolor="black", alpha=0.7)
            ax.set_title(f"{name.upper()} Distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.axvline(np.mean(values), color="red", linestyle="--", 
                      label=f"Mean: {np.mean(values):.4f}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # Ẩn axes thừa
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("TTS Evaluation - Score Distributions", fontsize=14)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "distribution.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Distribution plot saved to: {output_path}")
        return output_path

    def plot_radar_chart(self, results: List[MetricResult], output_dir: str) -> str:
        """Tạo radar chart hiển thị tổng quan các metric (normalized to [0,1]).

        Normalization:
        - MCD: lower is better, normalize inversely (1 - value/max_value)
        - PESQ: range [-0.5, 4.5], normalize to [0, 1]
        - STOI: already in [0, 1]
        - UTMOS: range [1, 5], normalize to [0, 1]
        - F0 Correlation: range [-1, 1], normalize to [0, 1]
        - WER: lower is better, normalize inversely (1 - value/max_value)

        Args:
            results: Danh sách MetricResult chứa kết quả đánh giá.
            output_dir: Thư mục lưu biểu đồ.

        Returns:
            Đường dẫn file PNG đã lưu.
        """
        os.makedirs(output_dir, exist_ok=True)

        metric_values = self._extract_metric_values(results)

        # Normalize metrics to [0, 1] scale
        normalized: Dict[str, float] = {}
        for name, values in metric_values.items():
            if not values:
                continue
            mean_val = float(np.mean(values))
            normalized[name] = self._normalize_metric(name, mean_val)

        if not normalized:
            logger.warning("No metric data available for radar chart.")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title("TTS Evaluation - Radar Chart (No Data)")
            output_path = os.path.join(output_dir, "radar_chart.png")
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        labels = [name.upper() for name in normalized.keys()]
        values = list(normalized.values())

        # Đóng radar chart
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_closed = values + [values[0]]
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles_closed, values_closed, "o-", linewidth=2, color="steelblue")
        ax.fill(angles_closed, values_closed, alpha=0.25, color="steelblue")
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("TTS Evaluation - Radar Chart (Normalized)", 
                    fontsize=14, pad=20)
        ax.grid(True)

        output_path = os.path.join(output_dir, "radar_chart.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Radar chart saved to: {output_path}")
        return output_path

    def plot_comparison(
        self, results_dict: Dict[str, List[MetricResult]], output_dir: str
    ) -> str:
        """Tạo comparison plot so sánh nhiều model trên từng metric.

        Args:
            results_dict: Dict mapping model_name -> List[MetricResult].
            output_dir: Thư mục lưu biểu đồ.

        Returns:
            Đường dẫn file PNG đã lưu.
        """
        os.makedirs(output_dir, exist_ok=True)

        model_names = list(results_dict.keys())

        # Tính mean cho từng metric của từng model
        model_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        for model_name, results in results_dict.items():
            metric_values = self._extract_metric_values(results)
            model_metrics[model_name] = {}
            for name, values in metric_values.items():
                if values:
                    model_metrics[model_name][name] = float(np.mean(values))
                else:
                    model_metrics[model_name][name] = None

        # Tìm metric có dữ liệu ở ít nhất 1 model
        active_metrics = []
        for name in METRIC_NAMES:
            if any(
                model_metrics[m].get(name) is not None for m in model_names
            ):
                active_metrics.append(name)

        if not active_metrics:
            logger.warning("No metric data available for comparison plot.")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("TTS Evaluation - Model Comparison (No Data)")
            output_path = os.path.join(output_dir, "comparison.png")
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        x = np.arange(len(active_metrics))
        width = 0.8 / len(model_names)
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model_name in enumerate(model_names):
            values = []
            for name in active_metrics:
                val = model_metrics[model_name].get(name)
                values.append(val if val is not None else 0.0)
            offset = (i - len(model_names) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model_name, 
                  color=colors[i], edgecolor="black", alpha=0.8)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Mean Value")
        ax.set_title("TTS Evaluation - Model Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in active_metrics])
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "comparison.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Comparison plot saved to: {output_path}")
        return output_path

    def plot_from_result_files(
        self, result_file_paths: List[str], output_dir: str
    ) -> List[str]:
        """Đọc ResultFile JSON và tạo comparison plots.

        Args:
            result_file_paths: Danh sách đường dẫn đến các file ResultFile JSON.
            output_dir: Thư mục lưu biểu đồ.

        Returns:
            Danh sách đường dẫn các file biểu đồ đã tạo.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Đọc và parse các ResultFile
        results_by_model: Dict[str, List[MetricResult]] = {}

        for file_path in result_file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"ResultFile not found, skipping: {file_path}")
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                model_name = data.get("model_name", "unknown")
                metric_name = data.get("metric_name", "")
                samples = data.get("samples", [])

                if model_name not in results_by_model:
                    results_by_model[model_name] = []

                # Chuyển đổi samples thành MetricResult
                for sample in samples:
                    sample_id = sample.get("sample_id", "")
                    text = sample.get("text", "")
                    value = sample.get("value")

                    if value is None:
                        continue

                    # Tạo MetricResult với metric tương ứng
                    kwargs = {
                        "sample_id": sample_id,
                        "text": text,
                    }
                    if metric_name in METRIC_NAMES:
                        kwargs[metric_name] = value

                    result = MetricResult(**kwargs)

                    # Tìm existing result cho sample_id này và merge
                    existing = None
                    for r in results_by_model[model_name]:
                        if r.sample_id == sample_id:
                            existing = r
                            break

                    if existing is not None:
                        # Merge metric value vào existing result
                        if metric_name in METRIC_NAMES:
                            setattr(existing, metric_name, value)
                    else:
                        results_by_model[model_name].append(result)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(
                    f"Invalid ResultFile format, skipping: {file_path}. Error: {e}"
                )
                continue

        if not results_by_model:
            logger.warning("No valid ResultFile data found.")
            return []

        chart_paths: List[str] = []

        # Tạo comparison plot nếu có nhiều model
        if len(results_by_model) > 1:
            path = self.plot_comparison(results_by_model, output_dir)
            chart_paths.append(path)

        # Tạo bar chart và radar chart cho từng model
        for model_name, results in results_by_model.items():
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            bar_path = self.plot_bar_chart(results, model_dir)
            chart_paths.append(bar_path)

            radar_path = self.plot_radar_chart(results, model_dir)
            chart_paths.append(radar_path)

        return chart_paths

    def plot_results(self, results: List[MetricResult], output_dir: str) -> List[str]:
        """Convenience method tạo tất cả biểu đồ cơ bản.

        Gọi bar_chart, distribution, và radar_chart.

        Args:
            results: Danh sách MetricResult chứa kết quả đánh giá.
            output_dir: Thư mục lưu biểu đồ.

        Returns:
            Danh sách đường dẫn các file biểu đồ đã tạo.
        """
        os.makedirs(output_dir, exist_ok=True)

        chart_paths: List[str] = []

        bar_path = self.plot_bar_chart(results, output_dir)
        chart_paths.append(bar_path)

        dist_path = self.plot_distribution(results, output_dir)
        chart_paths.append(dist_path)

        radar_path = self.plot_radar_chart(results, output_dir)
        chart_paths.append(radar_path)

        logger.info(f"All charts saved to: {output_dir}")
        return chart_paths

    def compare_from_folder(
        self, results_dir: str, output_dir: str
    ) -> List[str]:
        """Đọc tất cả ResultFile trong thư mục và tạo báo cáo so sánh.

        Tự động nhóm các file theo model và metric dựa trên quy ước
        đặt tên `<MODEL_NAME>_<METRIC>.json`.

        Args:
            results_dir: Thư mục chứa các ResultFile JSON.
            output_dir: Thư mục lưu biểu đồ so sánh.

        Returns:
            Danh sách đường dẫn các file biểu đồ đã tạo.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Tìm tất cả file JSON trong thư mục
        result_files: List[str] = []
        if not os.path.exists(results_dir):
            logger.warning(f"Results directory not found: {results_dir}")
            return []

        for filename in os.listdir(results_dir):
            if filename.endswith(".json") and filename != "run_history.json":
                file_path = os.path.join(results_dir, filename)
                result_files.append(file_path)

        if not result_files:
            logger.warning(f"No ResultFile found in: {results_dir}")
            return []

        logger.info(
            f"Found {len(result_files)} ResultFiles in {results_dir}"
        )

        # Đọc và nhóm theo model/metric
        results_by_model: Dict[str, List[MetricResult]] = {}

        for file_path in result_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                model_name = data.get("model_name", "unknown")
                metric_name = data.get("metric_name", "")
                samples = data.get("samples", [])

                if not metric_name or metric_name not in METRIC_NAMES:
                    logger.warning(
                        f"Invalid metric_name in {file_path}, skipping"
                    )
                    continue

                if model_name not in results_by_model:
                    results_by_model[model_name] = []

                for sample in samples:
                    sample_id = sample.get("sample_id", "")
                    text = sample.get("text", "")
                    value = sample.get("value")

                    if value is None:
                        continue

                    # Tìm existing result cho sample_id này
                    existing = None
                    for r in results_by_model[model_name]:
                        if r.sample_id == sample_id:
                            existing = r
                            break

                    if existing is not None:
                        setattr(existing, metric_name, value)
                    else:
                        kwargs = {
                            "sample_id": sample_id,
                            "text": text,
                            metric_name: value,
                        }
                        results_by_model[model_name].append(
                            MetricResult(**kwargs)
                        )

            except (json.JSONDecodeError, KeyError, TypeError, IOError) as e:
                logger.warning(
                    f"Invalid ResultFile, skipping: {file_path}. Error: {e}"
                )
                continue

        if not results_by_model:
            logger.warning("No valid data found for comparison.")
            return []

        chart_paths: List[str] = []

        # Tạo comparison plot nếu có nhiều model
        if len(results_by_model) > 1:
            path = self.plot_comparison(results_by_model, output_dir)
            chart_paths.append(path)

        # Tạo radar chart cho từng model
        for model_name, results in results_by_model.items():
            model_dir = os.path.join(output_dir, model_name)
            radar_path = self.plot_radar_chart(results, model_dir)
            chart_paths.append(radar_path)

        logger.info(
            f"Comparison charts created: {len(chart_paths)} files"
        )
        return chart_paths

    @staticmethod
    def _normalize_metric(name: str, value: float) -> float:
        """Normalize metric value to [0, 1] scale.

        Args:
            name: Tên metric.
            value: Giá trị trung bình của metric.

        Returns:
            Giá trị normalized trong [0, 1].
        """
        if name == "mcd":
            # MCD: lower is better, typical range [0, ~15]
            # Normalize inversely: higher normalized = better
            normalized = max(0.0, 1.0 - value / 15.0)
        elif name == "pesq":
            # PESQ: range [-0.5, 4.5]
            normalized = (value + 0.5) / 5.0
        elif name == "stoi":
            # STOI: already in [0, 1]
            normalized = value
        elif name == "utmos":
            # UTMOS: range [1, 5]
            normalized = (value - 1.0) / 4.0
        elif name == "f0_correlation":
            # F0 Correlation: range [-1, 1]
            normalized = (value + 1.0) / 2.0
        elif name == "wer":
            # WER: lower is better, typical range [0, 1+]
            # Normalize inversely: higher normalized = better
            normalized = max(0.0, 1.0 - value)
        else:
            normalized = value

        # Clamp to [0, 1]
        return float(np.clip(normalized, 0.0, 1.0))
