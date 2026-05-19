"""
Unit tests cho MetricCalculator.compute_all.

Kiểm tra fault tolerance: mỗi metric lỗi không ảnh hưởng metric khác.
Sử dụng mock để tránh phụ thuộc vào các thư viện nặng (pesq, torch, etc.).
"""

import sys
import importlib
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

sys.path.insert(0, r"d:\CO_2026\TTS-DATA")

# Mock các module nặng trước khi import calculator
# Tạo mock cho evaluate.metrics submodules để tránh import thực
_mock_modules = {
    "pesq": MagicMock(),
    "pystoi": MagicMock(),
    "torch": MagicMock(),
    "librosa": MagicMock(),
    "scipy": MagicMock(),
    "scipy.stats": MagicMock(),
    "jiwer": MagicMock(),
    "transformers": MagicMock(),
}

for mod_name, mock_mod in _mock_modules.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock_mod

# Now we can import the calculator module
from evaluate.metrics.calculator import MetricCalculator
from evaluate.models import MetricResult


class TestMetricCalculatorComputeAll:
    """Test compute_all method of MetricCalculator."""

    @patch("evaluate.metrics.calculator.compute_mcd", return_value=5.0)
    @patch("evaluate.metrics.calculator.compute_pesq", return_value=3.5)
    @patch("evaluate.metrics.calculator.compute_stoi", return_value=0.85)
    @patch("evaluate.metrics.calculator.predict_mos", return_value=4.0)
    @patch("evaluate.metrics.calculator.compute_f0_correlation", return_value=0.9)
    @patch("evaluate.metrics.calculator.compute_wer", return_value=0.1)
    def test_compute_all_success(
        self, mock_wer, mock_f0, mock_mos, mock_stoi, mock_pesq, mock_mcd
    ):
        """Tất cả metric thành công -> MetricResult đầy đủ."""
        calc = MetricCalculator()
        ref = np.zeros(16000, dtype=np.float32)
        syn = np.zeros(16000, dtype=np.float32)

        result = calc.compute_all(ref, syn, sr=16000, text="xin chào", sample_id="s001")

        assert isinstance(result, MetricResult)
        assert result.sample_id == "s001"
        assert result.text == "xin chào"
        assert result.mcd == 5.0
        assert result.pesq == 3.5
        assert result.stoi == 0.85
        assert result.utmos == 4.0
        assert result.f0_correlation == 0.9
        assert result.wer == 0.1

    @patch("evaluate.metrics.calculator.compute_mcd", side_effect=RuntimeError("MCD failed"))
    @patch("evaluate.metrics.calculator.compute_pesq", return_value=3.5)
    @patch("evaluate.metrics.calculator.compute_stoi", return_value=0.85)
    @patch("evaluate.metrics.calculator.predict_mos", return_value=4.0)
    @patch("evaluate.metrics.calculator.compute_f0_correlation", return_value=0.9)
    @patch("evaluate.metrics.calculator.compute_wer", return_value=0.1)
    def test_mcd_failure_isolated(
        self, mock_wer, mock_f0, mock_mos, mock_stoi, mock_pesq, mock_mcd
    ):
        """MCD lỗi -> mcd=None, các metric khác vẫn có giá trị."""
        calc = MetricCalculator()
        ref = np.zeros(16000, dtype=np.float32)
        syn = np.zeros(16000, dtype=np.float32)

        result = calc.compute_all(ref, syn, sr=16000, text="test", sample_id="s002")

        assert result.mcd is None
        assert result.pesq == 3.5
        assert result.stoi == 0.85
        assert result.utmos == 4.0
        assert result.f0_correlation == 0.9
        assert result.wer == 0.1

    @patch("evaluate.metrics.calculator.compute_mcd", side_effect=RuntimeError("fail"))
    @patch("evaluate.metrics.calculator.compute_pesq", side_effect=RuntimeError("fail"))
    @patch("evaluate.metrics.calculator.compute_stoi", side_effect=RuntimeError("fail"))
    @patch("evaluate.metrics.calculator.predict_mos", side_effect=RuntimeError("fail"))
    @patch("evaluate.metrics.calculator.compute_f0_correlation", side_effect=RuntimeError("fail"))
    @patch("evaluate.metrics.calculator.compute_wer", side_effect=RuntimeError("fail"))
    def test_all_metrics_fail(
        self, mock_wer, mock_f0, mock_mos, mock_stoi, mock_pesq, mock_mcd
    ):
        """Tất cả metric lỗi -> MetricResult với tất cả None, không raise exception."""
        calc = MetricCalculator()
        ref = np.zeros(16000, dtype=np.float32)
        syn = np.zeros(16000, dtype=np.float32)

        result = calc.compute_all(ref, syn, sr=16000, text="test", sample_id="s003")

        assert result.sample_id == "s003"
        assert result.text == "test"
        assert result.mcd is None
        assert result.pesq is None
        assert result.stoi is None
        assert result.utmos is None
        assert result.f0_correlation is None
        assert result.wer is None

    @patch("evaluate.metrics.calculator.compute_mcd", return_value=5.0)
    @patch("evaluate.metrics.calculator.compute_pesq", return_value=3.5)
    @patch("evaluate.metrics.calculator.compute_stoi", side_effect=ValueError("stoi error"))
    @patch("evaluate.metrics.calculator.predict_mos", return_value=4.0)
    @patch("evaluate.metrics.calculator.compute_f0_correlation", side_effect=Exception("f0 error"))
    @patch("evaluate.metrics.calculator.compute_wer", return_value=0.2)
    def test_partial_failure(
        self, mock_wer, mock_f0, mock_mos, mock_stoi, mock_pesq, mock_mcd
    ):
        """Một số metric lỗi -> chỉ metric lỗi là None, còn lại có giá trị."""
        calc = MetricCalculator()
        ref = np.zeros(16000, dtype=np.float32)
        syn = np.zeros(16000, dtype=np.float32)

        result = calc.compute_all(ref, syn, sr=16000, text="hello", sample_id="s004")

        assert result.mcd == 5.0
        assert result.pesq == 3.5
        assert result.stoi is None
        assert result.utmos == 4.0
        assert result.f0_correlation is None
        assert result.wer == 0.2

    @patch("evaluate.metrics.calculator.compute_mcd", return_value=5.0)
    @patch("evaluate.metrics.calculator.compute_pesq", return_value=3.5)
    @patch("evaluate.metrics.calculator.compute_stoi", return_value=0.85)
    @patch("evaluate.metrics.calculator.predict_mos", return_value=4.0)
    @patch("evaluate.metrics.calculator.compute_f0_correlation", return_value=0.9)
    @patch("evaluate.metrics.calculator.compute_wer", return_value=0.1)
    def test_default_sample_id(
        self, mock_wer, mock_f0, mock_mos, mock_stoi, mock_pesq, mock_mcd
    ):
        """sample_id mặc định là chuỗi rỗng nếu không truyền."""
        calc = MetricCalculator()
        ref = np.zeros(16000, dtype=np.float32)
        syn = np.zeros(16000, dtype=np.float32)

        result = calc.compute_all(ref, syn, sr=16000, text="test")

        assert result.sample_id == ""
        assert result.text == "test"
