"""Tests cho ReportGenerator."""

import csv
import os
import tempfile

import pytest

from evaluate.models import MetricResult, EvaluationReport
from evaluate.report_generator import ReportGenerator


@pytest.fixture
def sample_results():
    """Tạo danh sách MetricResult mẫu cho testing."""
    return [
        MetricResult(
            sample_id="sample_001",
            text="Xin chào thế giới",
            mcd=5.2,
            pesq=3.5,
            stoi=0.85,
            utmos=4.0,
            f0_correlation=0.9,
            wer=0.1,
        ),
        MetricResult(
            sample_id="sample_002",
            text="Hôm nay trời đẹp",
            mcd=6.1,
            pesq=2.8,
            stoi=0.72,
            utmos=3.5,
            f0_correlation=0.75,
            wer=0.2,
        ),
        MetricResult(
            sample_id="sample_003",
            text="Tôi yêu Việt Nam",
            mcd=4.8,
            pesq=4.0,
            stoi=0.91,
            utmos=4.5,
            f0_correlation=0.95,
            wer=0.05,
        ),
    ]


@pytest.fixture
def results_with_none():
    """Tạo danh sách MetricResult có giá trị None."""
    return [
        MetricResult(
            sample_id="sample_001",
            text="Text 1",
            mcd=5.0,
            pesq=None,
            stoi=0.8,
            utmos=4.0,
            f0_correlation=None,
            wer=0.1,
        ),
        MetricResult(
            sample_id="sample_002",
            text="Text 2",
            mcd=7.0,
            pesq=3.0,
            stoi=None,
            utmos=3.0,
            f0_correlation=0.8,
            wer=None,
        ),
    ]


@pytest.fixture
def generator():
    """Tạo instance ReportGenerator."""
    return ReportGenerator()


class TestExportCSV:
    """Tests cho phương thức export_csv."""

    def test_export_csv_creates_file(self, generator, sample_results):
        """CSV file được tạo thành công."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            result = generator.export_csv(sample_results, output_path)

            assert os.path.exists(result)
            assert result == output_path

    def test_export_csv_correct_columns(self, generator, sample_results):
        """CSV có đúng các cột yêu cầu."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            generator.export_csv(sample_results, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)

            expected = ["sample_id", "text", "mcd", "pesq", "stoi", "utmos", "f0_correlation", "wer"]
            assert header == expected

    def test_export_csv_correct_row_count(self, generator, sample_results):
        """CSV có đúng số dòng dữ liệu."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            generator.export_csv(sample_results, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # 1 header + 3 data rows
            assert len(rows) == 4

    def test_export_csv_none_values(self, generator, results_with_none):
        """CSV xử lý giá trị None đúng cách."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            generator.export_csv(results_with_none, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                row1 = next(reader)

            # pesq is None for sample_001
            assert row1[3] == ""  # None becomes empty string in CSV

    def test_export_csv_creates_parent_directory(self, generator, sample_results):
        """Tạo thư mục cha nếu chưa tồn tại."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subdir", "nested", "results.csv")
            result = generator.export_csv(sample_results, output_path)

            assert os.path.exists(result)

    def test_export_csv_returns_output_path(self, generator, sample_results):
        """Trả về đúng output_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            result = generator.export_csv(sample_results, output_path)
            assert result == output_path


class TestComputeSummary:
    """Tests cho phương thức compute_summary."""

    def test_compute_summary_all_metrics(self, generator, sample_results):
        """Tính summary cho tất cả metric."""
        summary = generator.compute_summary(sample_results)

        assert "mcd" in summary
        assert "pesq" in summary
        assert "stoi" in summary
        assert "utmos" in summary
        assert "f0_correlation" in summary
        assert "wer" in summary

    def test_compute_summary_has_required_stats(self, generator, sample_results):
        """Mỗi metric có mean, std, min, max."""
        summary = generator.compute_summary(sample_results)

        for metric_name, stats in summary.items():
            assert "mean" in stats, f"{metric_name} thiếu mean"
            assert "std" in stats, f"{metric_name} thiếu std"
            assert "min" in stats, f"{metric_name} thiếu min"
            assert "max" in stats, f"{metric_name} thiếu max"

    def test_compute_summary_correct_values(self, generator, sample_results):
        """Kiểm tra giá trị tính toán chính xác."""
        summary = generator.compute_summary(sample_results)

        # MCD values: 5.2, 6.1, 4.8
        mcd_stats = summary["mcd"]
        assert abs(mcd_stats["min"] - 4.8) < 1e-9
        assert abs(mcd_stats["max"] - 6.1) < 1e-9
        expected_mean = (5.2 + 6.1 + 4.8) / 3
        assert abs(mcd_stats["mean"] - expected_mean) < 1e-9

    def test_compute_summary_skips_none(self, generator, results_with_none):
        """Bỏ qua giá trị None khi tính summary."""
        summary = generator.compute_summary(results_with_none)

        # pesq: only one value (3.0) from sample_002
        assert "pesq" in summary
        assert abs(summary["pesq"]["mean"] - 3.0) < 1e-9

        # mcd: values 5.0, 7.0
        assert abs(summary["mcd"]["mean"] - 6.0) < 1e-9

    def test_compute_summary_empty_results(self, generator):
        """Trả về dict rỗng khi không có kết quả."""
        summary = generator.compute_summary([])
        assert summary == {}

    def test_compute_summary_all_none_metric(self, generator):
        """Metric toàn None không xuất hiện trong summary."""
        results = [
            MetricResult(sample_id="s1", text="t1", mcd=None, pesq=None),
            MetricResult(sample_id="s2", text="t2", mcd=None, pesq=3.0),
        ]
        summary = generator.compute_summary(results)
        assert "mcd" not in summary
        assert "pesq" in summary


class TestGenerateReport:
    """Tests cho phương thức generate_report."""

    def test_generate_report_returns_evaluation_report(self, generator, sample_results):
        """Trả về EvaluationReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generator.generate_report(sample_results, ["chart1.png"], tmpdir)
            assert isinstance(report, EvaluationReport)

    def test_generate_report_has_csv_path(self, generator, sample_results):
        """Report chứa csv_path hợp lệ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generator.generate_report(sample_results, [], tmpdir)
            assert report.csv_path is not None
            assert os.path.exists(report.csv_path)

    def test_generate_report_has_summary(self, generator, sample_results):
        """Report chứa summary_statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generator.generate_report(sample_results, [], tmpdir)
            assert len(report.summary_statistics) > 0
            assert "mcd" in report.summary_statistics

    def test_generate_report_has_chart_paths(self, generator, sample_results):
        """Report chứa chart_paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_paths = ["chart1.png", "chart2.png"]
            report = generator.generate_report(sample_results, chart_paths, tmpdir)
            assert report.chart_paths == chart_paths

    def test_generate_report_has_results(self, generator, sample_results):
        """Report chứa results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generator.generate_report(sample_results, [], tmpdir)
            assert report.results == sample_results
            assert len(report.results) == 3
