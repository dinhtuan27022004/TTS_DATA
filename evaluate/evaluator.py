"""
TTSEvaluator - Class điều phối chính cho hệ thống đánh giá TTS.

Tích hợp tất cả module: DatasetLoader, MetricCalculator, Visualizer,
ReportGenerator, Persistence, và ResumeManager để thực hiện toàn bộ
quy trình đánh giá từ tải dataset đến xuất báo cáo.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from evaluate.dataset_loader import DatasetLoader
from evaluate.metrics.calculator import MetricCalculator
from evaluate.models import (
    EvalSample,
    EvaluationReport,
    MetricResult,
    RunHistoryEntry,
    TTSModel,
)
from evaluate.persistence import (
    ensure_results_folder,
    load_result_file,
    save_result_file,
    save_run_history,
    sanitize_filename,
)
from evaluate.report_generator import ReportGenerator
from evaluate.resume import ResumeManager
from evaluate.visualizer import Visualizer

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_HISTORY_PATH = os.path.join(DEFAULT_RESULTS_DIR, "run_history.json")

METRIC_NAMES = ["mcd", "pesq", "stoi", "utmos", "f0_correlation", "wer", "cer"]


class TTSEvaluator:
    """Class điều phối chính cho hệ thống đánh giá TTS.

    Thực hiện toàn bộ quy trình đánh giá:
    1. Tải dataset
    2. Tổng hợp audio từ mô hình TTS
    3. Tính toán metric
    4. Lưu kết quả (với resume support)
    5. Tạo biểu đồ và báo cáo
    """

    def __init__(
        self,
        results_dir: str = DEFAULT_RESULTS_DIR,
        history_path: str = DEFAULT_HISTORY_PATH,
    ):
        """Khởi tạo TTSEvaluator.

        Args:
            results_dir: Thư mục lưu kết quả JSON.
            history_path: Đường dẫn file RunHistory.
        """
        self.results_dir = results_dir
        self.history_path = history_path
        self.dataset_loader = DatasetLoader()
        self.metric_calculator = MetricCalculator()
        self.visualizer = Visualizer()
        self.report_generator = ReportGenerator()
        self.resume_manager = ResumeManager()

        ensure_results_folder(results_dir)

    def evaluate(
        self,
        dataset_path: str,
        tts_model: TTSModel,
        model_name: str = "unknown",
        metrics: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        force: bool = False,
        save_audio: bool = False,
        audio_output_dir: Optional[str] = None,
    ) -> EvaluationReport:
        """Thực hiện toàn bộ quy trình đánh giá TTS.

        Args:
            dataset_path: Đường dẫn đến thư mục dataset.
            tts_model: Mô hình TTS cần đánh giá (implement TTSModel).
            model_name: Tên mô hình (dùng cho lưu file và báo cáo).
            metrics: Danh sách metric cần tính. None = tất cả.
            output_dir: Thư mục xuất báo cáo. None = results_dir.

        Returns:
            EvaluationReport chứa đầy đủ kết quả.

        Raises:
            FileNotFoundError: Khi dataset_path không hợp lệ.
            RuntimeError: Khi không có mẫu nào được đánh giá thành công.
        """
        if output_dir is None:
            output_dir = self.results_dir

        if metrics is None:
            metrics = METRIC_NAMES

        # Setup audio output directory
        if save_audio:
            if audio_output_dir is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                audio_output_dir = os.path.join(base_dir, "artifact", sanitize_filename(model_name))
            os.makedirs(audio_output_dir, exist_ok=True)
            logger.info(f"Saving synthesized audio to: {audio_output_dir}")

        # 1. Tải dataset
        logger.info(f"Loading dataset from: {dataset_path}")
        samples = self.dataset_loader.load_dataset(dataset_path)
        logger.info(f"Loaded {len(samples)} samples")

        if not samples:
            raise RuntimeError(
                f"No valid samples found in dataset: {dataset_path}"
            )

        # 2. Tổng hợp audio và tính metric cho từng mẫu
        all_results: List[MetricResult] = []
        metric_data: Dict[str, List[Dict]] = {m: [] for m in metrics}

        # Kiểm tra resume cho từng metric
        if not force:
            pass  # Resume logic đã chuyển xuống phần xác định pending_samples
        else:
            # Xóa các ResultFile cũ khi force=True
            for metric_name in metrics:
                safe_name = sanitize_filename(model_name)
                old_file = os.path.join(self.results_dir, f"{safe_name}_{metric_name}.json")
                if os.path.exists(old_file):
                    os.remove(old_file)
                    logger.info(f"Removed old result file: {old_file}")

        # Xác định mẫu cần xử lý
        pending_samples = samples  # Mặc định xử lý tất cả

        if not force:
            # Resume: kiểm tra kết quả đã có và chỉ xử lý mẫu chưa tính
            # Dùng metric đầu tiên làm chuẩn (tất cả metric được tính cùng lúc)
            first_metric = metrics[0]
            existing = self.resume_manager.check_existing_results(
                model_name, first_metric, self.results_dir
            )
            if existing:
                pending_samples = self.resume_manager.get_pending_samples(
                    samples, existing
                )
                # Load kết quả cũ vào metric_data
                existing_samples = existing.get("samples", [])
                for metric_name in metrics:
                    existing_metric = self.resume_manager.check_existing_results(
                        model_name, metric_name, self.results_dir
                    )
                    if existing_metric:
                        metric_data[metric_name] = existing_metric.get("samples", [])

        # Tổng hợp và tính metric
        logger.info(f"Processing {len(pending_samples)} samples (skipped {len(samples) - len(pending_samples)} already computed)...")
        save_interval = 10  # Lưu kết quả mỗi 10 samples
        
        for idx, sample in enumerate(tqdm(pending_samples, desc="Evaluating")):
            try:
                # Tổng hợp audio (truyền ref_audio_path và ref_text từ dataset)
                syn_audio, syn_sr = tts_model.synthesize(
                    gen_text=sample.text,
                    ref_audio_path=sample.audio_path,
                    ref_text=sample.text,
                )
            except Exception as e:
                logger.error(
                    f"TTS synthesis failed for sample {sample.sample_id}: {e}"
                )
                continue

            # Lưu audio tổng hợp nếu được yêu cầu
            if save_audio and audio_output_dir:
                import soundfile as sf
                audio_filename = os.path.basename(sample.audio_path)
                audio_save_path = os.path.join(audio_output_dir, audio_filename)
                sf.write(audio_save_path, syn_audio, syn_sr)

            try:
                # Tải ground truth audio
                ref_audio, ref_sr = self.dataset_loader.load_audio(
                    sample.audio_path
                )
            except Exception as e:
                logger.error(
                    f"Failed to load reference audio for {sample.sample_id}: {e}"
                )
                continue

            # Tính tất cả metric
            result = self.metric_calculator.compute_all(
                ref_audio=ref_audio,
                syn_audio=syn_audio,
                sr=ref_sr,
                text=sample.text,
                sample_id=sample.sample_id,
            )
            all_results.append(result)

            # Thu thập dữ liệu cho từng metric
            for metric_name in metrics:
                value = getattr(result, metric_name, None)
                sample_entry = {
                    "sample_id": sample.sample_id,
                    "value": value,
                    "text": sample.text,
                }
                # Thêm transcription cho metric WER
                if metric_name == "wer" and result.transcription is not None:
                    sample_entry["transcription"] = result.transcription
                metric_data[metric_name].append(sample_entry)

            # Lưu incremental mỗi save_interval samples
            if (idx + 1) % save_interval == 0:
                self._save_intermediate_results(
                    model_name, metrics, metric_data
                )

        # 3. Kiểm tra có kết quả không
        if not all_results:
            raise RuntimeError(
                "No samples were evaluated successfully. "
                "All samples failed during synthesis or metric computation."
            )

        logger.info(f"Successfully evaluated {len(all_results)} samples")

        # 4. Lưu ResultFile cho từng metric
        for metric_name in metrics:
            samples_data = metric_data[metric_name]
            if not samples_data:
                continue

            # Tính summary cho metric này
            values = [
                s["value"] for s in samples_data if s["value"] is not None
            ]
            summary = {}
            if values:
                summary = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                }

            save_result_file(
                model_name=model_name,
                metric_name=metric_name,
                samples=samples_data,
                summary=summary,
                output_dir=self.results_dir,
            )

            # Lưu RunHistory
            start_time = datetime.now().isoformat()
            entry = RunHistoryEntry(
                model_name=model_name,
                metric_name=metric_name,
                start_time=start_time,
                end_time=datetime.now().isoformat(),
                num_samples=len(samples_data),
                status="completed",
            )
            save_run_history(entry, self.history_path)

        # 5. Tạo biểu đồ
        charts_dir = os.path.join(output_dir, "charts")
        chart_paths = self.visualizer.plot_results(all_results, charts_dir)

        # 6. Tạo báo cáo
        report = self.report_generator.generate_report(
            results=all_results,
            chart_paths=chart_paths,
            output_dir=output_dir,
        )

        logger.info(
            f"Evaluation complete: {len(all_results)} samples, "
            f"{len(chart_paths)} charts, CSV at {report.csv_path}"
        )

        return report

    def _save_intermediate_results(
        self,
        model_name: str,
        metrics: List[str],
        metric_data: Dict[str, List[Dict]],
    ) -> None:
        """Lưu kết quả trung gian vào ResultFile (incremental save).

        Args:
            model_name: Tên model.
            metrics: Danh sách metric.
            metric_data: Dữ liệu metric đã thu thập.
        """
        for metric_name in metrics:
            samples_data = metric_data[metric_name]
            if not samples_data:
                continue

            values = [
                s["value"] for s in samples_data if s["value"] is not None
            ]
            summary = {}
            if values:
                summary = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                }

            save_result_file(
                model_name=model_name,
                metric_name=metric_name,
                samples=samples_data,
                summary=summary,
                output_dir=self.results_dir,
            )
