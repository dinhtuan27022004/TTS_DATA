"""
Module xuất báo cáo kết quả đánh giá TTS.

Chứa class ReportGenerator với các chức năng:
- Xuất kết quả chi tiết ra file CSV
- Tính toán summary statistics (mean, std, min, max) cho từng metric
- Tạo EvaluationReport tổng hợp
"""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from evaluate.models import EvaluationReport, MetricResult

logger = logging.getLogger(__name__)

# Danh sách tên các metric tương ứng với thuộc tính trong MetricResult
METRIC_NAMES = ["mcd", "pesq", "stoi", "utmos", "f0_correlation", "wer", "cer"]

# Các cột trong file CSV xuất ra
CSV_COLUMNS = ["sample_id", "text"] + METRIC_NAMES


class ReportGenerator:
    """Xuất báo cáo kết quả đánh giá TTS.

    Cung cấp các phương thức để:
    - Xuất file CSV chứa kết quả chi tiết từng mẫu
    - Tính toán summary statistics cho từng metric
    - Tạo EvaluationReport đầy đủ
    """

    def export_csv(self, results: List[MetricResult], output_path: str) -> str:
        """Xuất kết quả đánh giá ra file CSV.

        Tạo file CSV với các cột: sample_id, text, mcd, pesq, stoi, utmos,
        f0_correlation, wer. Mỗi dòng tương ứng với một mẫu đánh giá.

        Args:
            results: Danh sách MetricResult chứa kết quả từng mẫu.
            output_path: Đường dẫn file CSV đầu ra.

        Returns:
            Đường dẫn file CSV đã lưu (chính là output_path).
        """
        # Tạo thư mục cha nếu chưa tồn tại
        parent_dir = Path(output_path).parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

            for result in results:
                row = [
                    result.sample_id,
                    result.text,
                    result.mcd,
                    result.pesq,
                    result.stoi,
                    result.utmos,
                    result.f0_correlation,
                    result.wer,
                ]
                writer.writerow(row)

        logger.info(f"Đã xuất CSV với {len(results)} mẫu tại: {output_path}")
        return output_path

    def compute_summary(
        self, results: List[MetricResult]
    ) -> Dict[str, Dict[str, float]]:
        """Tính toán summary statistics cho từng metric.

        Tính mean, std, min, max cho mỗi metric, bỏ qua các giá trị None.

        Args:
            results: Danh sách MetricResult chứa kết quả từng mẫu.

        Returns:
            Dict với key là tên metric, value là dict chứa mean, std, min, max.
            Ví dụ: {"mcd": {"mean": 5.2, "std": 1.1, "min": 3.0, "max": 8.5}, ...}
            Metric không có giá trị hợp lệ nào sẽ không xuất hiện trong kết quả.
        """
        summary: Dict[str, Dict[str, float]] = {}

        for metric_name in METRIC_NAMES:
            # Thu thập các giá trị không None cho metric này
            values = [
                getattr(result, metric_name)
                for result in results
                if getattr(result, metric_name) is not None
            ]

            if not values:
                logger.warning(
                    f"Metric '{metric_name}' không có giá trị hợp lệ nào, bỏ qua."
                )
                continue

            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            min_val = float(np.min(values))
            max_val = float(np.max(values))

            summary[metric_name] = {
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
            }

        logger.info(f"Đã tính summary statistics cho {len(summary)} metric.")
        return summary

    def generate_report(
        self,
        results: List[MetricResult],
        chart_paths: List[str],
        output_dir: str,
    ) -> EvaluationReport:
        """Tạo EvaluationReport đầy đủ.

        Gọi export_csv để lưu CSV, compute_summary để tính thống kê,
        và trả về EvaluationReport chứa tất cả thông tin.

        Args:
            results: Danh sách MetricResult chứa kết quả từng mẫu.
            chart_paths: Danh sách đường dẫn đến các file biểu đồ đã tạo.
            output_dir: Thư mục đầu ra để lưu file CSV.

        Returns:
            EvaluationReport chứa csv_path, summary_statistics, chart_paths,
            và results.
        """
        # Xuất CSV
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        self.export_csv(results, csv_path)

        # Tính summary statistics
        summary_statistics = self.compute_summary(results)

        # Tạo và trả về EvaluationReport
        report = EvaluationReport(
            results=results,
            chart_paths=chart_paths,
            summary_statistics=summary_statistics,
            csv_path=csv_path,
        )

        logger.info(
            f"Đã tạo EvaluationReport: {len(results)} mẫu, "
            f"{len(chart_paths)} biểu đồ, CSV tại {csv_path}"
        )
        return report
