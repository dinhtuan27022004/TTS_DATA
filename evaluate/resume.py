"""
ResumeManager - Quản lý logic resume cho hệ thống đánh giá TTS.

Cho phép hệ thống tiếp tục đánh giá từ vị trí đã dừng thay vì chạy lại từ đầu.
Kiểm tra ResultFile đã tồn tại, xác định mẫu chưa có kết quả, và xử lý file lỗi.
"""

import logging
import os
import shutil
from typing import Dict, List, Optional, Set

from evaluate.models import EvalSample
from evaluate.persistence import load_result_file, sanitize_filename

logger = logging.getLogger(__name__)


class ResumeManager:
    """Quản lý logic resume cho quá trình đánh giá TTS.

    Kiểm tra kết quả đã tồn tại và xác định mẫu cần tính toán thêm.
    """

    def check_existing_results(
        self, model_name: str, metric_name: str, results_dir: str
    ) -> Optional[Dict]:
        """Kiểm tra xem ResultFile tương ứng đã tồn tại hay chưa.

        Args:
            model_name: Tên mô hình TTS.
            metric_name: Tên metric.
            results_dir: Thư mục chứa các ResultFile.

        Returns:
            Dict chứa dữ liệu ResultFile nếu tồn tại và hợp lệ, None nếu không.
        """
        safe_name = sanitize_filename(model_name)
        filename = f"{safe_name}_{metric_name}.json"
        file_path = os.path.join(results_dir, filename)

        if not os.path.exists(file_path):
            logger.info(
                f"No existing ResultFile for {model_name}/{metric_name}"
            )
            return None

        data = load_result_file(file_path)

        if data is None:
            # File tồn tại nhưng không hợp lệ -> handle corrupt
            self.handle_corrupt_file(file_path)
            return None

        logger.info(
            f"Found existing ResultFile for {model_name}/{metric_name}: "
            f"{len(data.get('samples', []))} samples"
        )
        return data

    def get_pending_samples(
        self,
        all_samples: List[EvalSample],
        existing_results: Optional[Dict],
    ) -> List[EvalSample]:
        """Trả về danh sách mẫu chưa có kết quả.

        So sánh danh sách mẫu cần đánh giá với kết quả đã có,
        trả về chỉ những mẫu chưa được tính toán.

        Args:
            all_samples: Danh sách tất cả EvalSample cần đánh giá.
            existing_results: Dữ liệu ResultFile đã tồn tại (hoặc None).

        Returns:
            Danh sách EvalSample chưa có kết quả.
        """
        if existing_results is None:
            return all_samples

        # Lấy set sample_id đã có kết quả
        existing_ids: Set[str] = set()
        for sample in existing_results.get("samples", []):
            sample_id = sample.get("sample_id", "")
            if sample_id:
                existing_ids.add(sample_id)

        # Lọc ra mẫu chưa có kết quả
        pending = [s for s in all_samples if s.sample_id not in existing_ids]

        skipped = len(all_samples) - len(pending)
        logger.info(
            f"Resume: {skipped} samples already computed, "
            f"{len(pending)} samples pending"
        )

        return pending

    def handle_corrupt_file(self, file_path: str) -> None:
        """Xử lý file ResultFile bị lỗi.

        Đổi tên file lỗi thành <tên_file>.backup để bảo toàn dữ liệu,
        sau đó hệ thống sẽ tính toán lại từ đầu.

        Args:
            file_path: Đường dẫn đến file bị lỗi.
        """
        if not os.path.exists(file_path):
            return

        backup_path = f"{file_path}.backup"

        # Nếu backup đã tồn tại, thêm số thứ tự
        counter = 1
        while os.path.exists(backup_path):
            backup_path = f"{file_path}.backup.{counter}"
            counter += 1

        try:
            shutil.move(file_path, backup_path)
            logger.warning(
                f"Corrupt ResultFile renamed to backup: "
                f"{file_path} -> {backup_path}"
            )
        except OSError as e:
            logger.error(
                f"Failed to rename corrupt file {file_path}: {e}"
            )
