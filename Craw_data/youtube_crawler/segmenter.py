"""
Phase 5: Audio Segmentation
Cắt audio thành đoạn 3-7 giây dựa trên word-level timestamps.
Đảm bảo không cắt giữa từ, không tạo segment trống.
"""

import os
import json
import logging
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .models import WordTimestamp, SegmentInfo, StatsInfo

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioSegmenter:
    """
    Cắt audio thành đoạn 3-7 giây dựa trên timestamps.
    
    - Đọc timestamps từ JSON (output Phase 4)
    - Tìm điểm cắt tại word boundaries (không cắt giữa từ)
    - Cắt audio thành đoạn 3-7 giây
    - Loại bỏ đoạn trống (không có speech)
    - Naming: {basename}_{00001}.wav/txt
    - Resume: skip file gốc đã có segments trong output
    - Lưu stats.json: total_files, total_duration, avg_duration
    """

    def __init__(
        self,
        input_dir: str = "Youtube_Data/Step_1",
        output_dir: str = "Youtube_Data/Step_2",
        min_duration: float = 3.0,
        max_duration: float = 7.0
    ):
        """
        Args:
            input_dir: Thư mục Step_1 chứa WAV + timestamps JSON
            output_dir: Thư mục Step_2 để lưu segments
            min_duration: Thời lượng tối thiểu mỗi segment (giây)
            max_duration: Thời lượng tối đa mỗi segment (giây)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)

        # Danh sách tất cả segments đã tạo (dùng cho stats)
        self.all_segments: List[SegmentInfo] = []

    def segment_all(self) -> List[SegmentInfo]:
        """
        Cắt tất cả file thành segments, resume từ vị trí dừng.
        
        - Liệt kê tất cả file .json trong input_dir (timestamps từ Phase 4)
        - Skip file gốc đã có segments trong output (resume)
        - Cắt audio và lưu segments
        - Lưu stats.json
        
        Returns:
            Danh sách SegmentInfo (path, transcript, duration)
        """
        # Liệt kê tất cả file JSON timestamps trong input_dir
        all_json_files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(".json")
        ]
        all_json_files.sort()

        if not all_json_files:
            logger.warning(f"Không tìm thấy file JSON nào trong {self.input_dir}")
            return []

        logger.info(f"Tìm thấy {len(all_json_files)} file JSON timestamps")

        # Kiểm tra file nào đã xử lý (resume)
        files_to_process = []
        for json_filename in all_json_files:
            basename = os.path.splitext(json_filename)[0]
            if self._is_already_segmented(basename):
                logger.info(f"[SKIP] Đã có segments: {basename}")
            else:
                files_to_process.append(json_filename)

        skipped = len(all_json_files) - len(files_to_process)
        logger.info(f"Đã xử lý trước đó: {skipped} files")
        logger.info(f"Cần xử lý thêm: {len(files_to_process)} files")

        if not files_to_process:
            logger.info("Tất cả file đã được segment. Không cần xử lý thêm.")
            # Vẫn lưu stats dựa trên file hiện có trong output
            self._save_stats()
            return []

        # Xử lý từng file
        new_segments = []
        for json_filename in tqdm(files_to_process, desc="Segmenting"):
            json_path = os.path.join(self.input_dir, json_filename)
            try:
                segments = self._process_json_file(json_path)
                new_segments.extend(segments)
                logger.info(f"[OK] {json_filename} -> {len(segments)} segments")
            except Exception as e:
                logger.error(f"[FAIL] Lỗi xử lý {json_filename}: {e}")

        # Lưu stats.json
        self._save_stats()

        logger.info(f"Hoàn thành! Tạo {len(new_segments)} segments mới.")
        return new_segments

    def _is_already_segmented(self, basename: str) -> bool:
        """
        Kiểm tra file gốc đã có segments trong output chưa (resume).
        
        Tìm file có pattern {basename}_00001.wav trong output_dir.
        Nếu tồn tại ít nhất 1 segment -> đã xử lý.
        
        Args:
            basename: Tên file gốc (không có extension)
            
        Returns:
            True nếu đã có segments
        """
        first_segment = os.path.join(self.output_dir, f"{basename}_00001.wav")
        return os.path.exists(first_segment)

    def _process_json_file(self, json_path: str) -> List[SegmentInfo]:
        """
        Đọc JSON timestamps và cắt audio tương ứng.
        
        Args:
            json_path: Đường dẫn file JSON timestamps
            
        Returns:
            Danh sách SegmentInfo
        """
        # Đọc JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse word timestamps
        timestamps = [
            WordTimestamp(
                word=wt["word"],
                start_time=wt["start_time"],
                end_time=wt["end_time"]
            )
            for wt in data.get("word_timestamps", [])
        ]

        if not timestamps:
            logger.warning(f"Không có word timestamps trong {json_path}")
            return []

        # Tìm file WAV tương ứng
        wav_path = data.get("wav_path", "")
        if not wav_path or not os.path.exists(wav_path):
            # Thử tìm WAV cùng tên trong input_dir
            basename = os.path.splitext(os.path.basename(json_path))[0]
            wav_path = os.path.join(self.input_dir, f"{basename}.wav")

        if not os.path.exists(wav_path):
            logger.warning(f"Không tìm thấy file WAV cho {json_path}")
            return []

        # Cắt file thành segments
        return self._segment_file(wav_path, timestamps)

    def _segment_file(self, wav_path: str, timestamps: List[WordTimestamp]) -> List[SegmentInfo]:
        """
        Cắt 1 file thành các đoạn 3-7s.
        
        - Tìm điểm cắt tối ưu tại word boundaries
        - Cắt audio và lưu WAV + TXT
        - Naming: {basename}_{00001}.wav/txt
        
        Args:
            wav_path: Đường dẫn file WAV gốc
            timestamps: Danh sách WordTimestamp
            
        Returns:
            Danh sách SegmentInfo
        """
        # Tìm các đoạn cắt tối ưu
        segments_data = self._find_cut_points(timestamps)

        if not segments_data:
            logger.warning(f"Không tìm được segment hợp lệ cho {wav_path}")
            return []

        # Đọc audio
        audio, sr = sf.read(wav_path)

        # Nếu audio stereo, convert sang mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Tên file gốc (không có extension)
        basename = Path(wav_path).stem

        # Cắt và lưu từng segment
        results = []
        for idx, (start_time, end_time, transcript) in enumerate(segments_data, start=1):
            # Tên file: basename_00001.wav
            segment_name = f"{basename}_{idx:05d}"
            wav_output = os.path.join(self.output_dir, f"{segment_name}.wav")
            txt_output = os.path.join(self.output_dir, f"{segment_name}.txt")

            # Cắt audio theo thời gian
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]

            # Kiểm tra segment có dữ liệu không
            if len(segment_audio) == 0:
                continue

            # Lưu WAV
            sf.write(wav_output, segment_audio, sr)

            # Lưu transcript TXT
            with open(txt_output, "w", encoding="utf-8") as f:
                f.write(transcript)

            # Tạo SegmentInfo
            duration = end_time - start_time
            segment_info = SegmentInfo(
                output_path=wav_output,
                transcript=transcript,
                duration=duration,
                source_file=wav_path,
                start_time=start_time,
                end_time=end_time
            )
            results.append(segment_info)

        return results

    def _find_cut_points(self, timestamps: List[WordTimestamp]) -> List[Tuple[float, float, str]]:
        """
        Tìm điểm cắt tối ưu dựa trên word boundaries và khoảng lặng (pauses).
        
        Thuật toán:
        - Duyệt qua từng word timestamp
        - Mở rộng segment cho đến khi đạt max_duration
        - Lùi lại (step back) từ điểm cuối để tìm khoảng lặng (gap > 0.05s) giữa các từ.
        - Nếu tìm thấy khoảng lặng và segment vẫn đủ min_duration, cắt tại đó.
        - Nếu không có khoảng lặng nào (đoạn nói liên tục), cắt tại điểm lớn nhất <= max_duration.
        - Nếu segment quá ngắn -> skip từ đầu, thử lại
        
        QUAN TRỌNG: Tránh cắt đôi một cụm từ liên tục (gap = 0.0) nếu có thể.
        
        Args:
            timestamps: Danh sách WordTimestamp đã sắp xếp theo thời gian
            
        Returns:
            Danh sách (start_time, end_time, transcript) cho mỗi segment
        """
        segments = []

        if not timestamps:
            return segments

        n = len(timestamps)
        i = 0  # Index từ hiện tại

        while i < n:
            segment_start = timestamps[i].start_time
            
            # Mở rộng segment cho đến khi đạt max_duration
            j = i
            while j < n:
                word_end = timestamps[j].end_time
                current_duration = word_end - segment_start

                # Dừng nếu thêm từ này sẽ vượt max_duration
                if current_duration > self.max_duration:
                    break
                    
                j += 1

            if j > i:
                best_cut_index = j - 1
                
                # Step back để tìm khoảng lặng (gap > 0.05s) tự nhiên
                candidate = best_cut_index
                while candidate >= i:
                    duration = timestamps[candidate].end_time - segment_start
                    if duration < self.min_duration:
                        break # Không thể lùi thêm vì sẽ vi phạm min_duration
                        
                    if candidate + 1 < n:
                        gap = timestamps[candidate+1].start_time - timestamps[candidate].end_time
                        if gap > 0.05: # Found a natural pause
                            best_cut_index = candidate
                            break
                    else:
                        break # Từ cuối cùng rồi
                    
                    candidate -= 1
                    
                segment_end = timestamps[best_cut_index].end_time
                duration = segment_end - segment_start
                
                segment_words = [t.word for t in timestamps[i:best_cut_index+1]]
                transcript = " ".join(segment_words)
                
                # Segment hợp lệ: đủ dài VÀ có transcript
                if duration >= self.min_duration and transcript.strip():
                    segments.append((segment_start, segment_end, transcript))
                    i = best_cut_index + 1
                else:
                    # Segment quá ngắn (có thể do step back thất bại hoặc do file quá ngắn)
                    i += 1
            else:
                # Không thêm được từ nào (từ đầu tiên đã vượt max_duration)
                i += 1

        return segments

    def _save_stats(self):
        """
        Lưu stats.json cho Step_2.
        
        Thống kê dựa trên tất cả file .wav hiện có trong output_dir.
        Format: {total_files, total_duration_seconds, avg_duration_seconds}
        """
        # Liệt kê tất cả file WAV trong output_dir
        wav_files = [
            f for f in os.listdir(self.output_dir)
            if f.lower().endswith(".wav")
        ]

        total_files = len(wav_files)
        total_duration = 0.0

        # Tính tổng duration từ tất cả file WAV
        for wav_filename in wav_files:
            wav_path = os.path.join(self.output_dir, wav_filename)
            try:
                info = sf.info(wav_path)
                total_duration += info.duration
            except Exception:
                pass

        # Tính average
        avg_duration = total_duration / total_files if total_files > 0 else 0.0

        # Tạo stats
        stats = {
            "total_files": total_files,
            "total_duration_seconds": round(total_duration, 2),
            "avg_duration_seconds": round(avg_duration, 2)
        }

        # Lưu stats.json
        stats_path = os.path.join(self.output_dir, "stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Stats: {total_files} files, {total_duration:.1f}s total, {avg_duration:.1f}s avg")
        logger.info(f"Đã lưu: {stats_path}")
