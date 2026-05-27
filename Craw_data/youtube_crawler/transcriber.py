"""
Phase 4: Transcription
Transcribe audio tiếng Việt bằng nvidia/parakeet-ctc-0.6b-vi.
Trích xuất word-level timestamps từ CTC model output.
"""

import os
import json
import logging
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .models import TranscriptionResult, WordTimestamp

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Transcriber:
    """
    Transcribe audio tiếng Việt bằng nvidia/parakeet-ctc-0.6b-vi.
    
    - Load model Parakeet CTC 0.6b-vi (NeMo)
    - Transcribe từng file WAV trong Step_1
    - Trích xuất word-level timestamps từ CTC alignment
    - Lưu timestamps dạng JSON (cùng tên file .json)
    - Resume: skip file đã có .json tương ứng
    """

    def __init__(
        self,
        input_dir: str = "Youtube_Data/Step_1",
        model_name: str = "nvidia/parakeet-ctc-0.6b-vi"
    ):
        """
        Args:
            input_dir: Thư mục Step_1 chứa vocals WAV
            model_name: Tên model NeMo ASR
        """
        self.input_dir = input_dir
        self.model_name = model_name

        # Model sẽ được load khi cần (lazy loading)
        self.model = None

    def _load_model(self):
        """
        Load model bằng NeMo toolkit hoặc WeNet toolkit.
        Tự động chọn GPU nếu có.
        """
        if "chunkformer" in self.model_name:
            import wenet
            logger.info(f"Đang load WeNet model từ: {self.model_name}...")
            # WeNet load_model handles CPU/GPU usage internally
            self.model = wenet.load_model(model_dir=self.model_name)
            logger.info(f"WeNet Model {self.model_name} đã load thành công!")
        else:
            import nemo.collections.asr as nemo_asr
            logger.info(f"Đang load NeMo model: {self.model_name}...")
            self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(self.model_name)

            # Chuyển sang GPU nếu có
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Sử dụng GPU (CUDA) cho transcription")
            else:
                logger.info("Sử dụng CPU cho transcription")

            self.model.eval()
            logger.info(f"Model {self.model_name} đã load thành công!")

    def transcribe_all(self) -> List[TranscriptionResult]:
        """
        Transcribe tất cả file WAV trong input_dir, resume từ vị trí dừng.
        
        - Liệt kê tất cả file .wav trong input_dir
        - Skip file đã có .json tương ứng (resume)
        - Transcribe và lưu timestamps JSON
        
        Returns:
            Danh sách TranscriptionResult với timestamps
        """
        # Liệt kê tất cả file WAV
        all_wav_files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(".wav")
        ]
        all_wav_files.sort()

        if not all_wav_files:
            logger.warning(f"Không tìm thấy file WAV nào trong {self.input_dir}")
            return []

        logger.info(f"Tìm thấy {len(all_wav_files)} file WAV trong {self.input_dir}")

        # Kiểm tra file nào đã có .json (resume)
        files_to_process = []
        for filename in all_wav_files:
            json_path = os.path.join(
                self.input_dir,
                os.path.splitext(filename)[0] + ".json"
            )
            if os.path.exists(json_path):
                logger.info(f"[SKIP] Đã có transcript: {filename}")
            else:
                files_to_process.append(filename)

        skipped = len(all_wav_files) - len(files_to_process)
        logger.info(f"Đã transcribe trước đó: {skipped} files")
        logger.info(f"Cần transcribe thêm: {len(files_to_process)} files")

        if not files_to_process:
            logger.info("Tất cả file đã được transcribe. Không cần xử lý thêm.")
            return []

        # Load model (chỉ load khi thực sự cần)
        if self.model is None:
            self._load_model()

        # Transcribe từng file
        results = []
        for filename in tqdm(files_to_process, desc="Transcribing"):
            wav_path = os.path.join(self.input_dir, filename)
            try:
                result = self._transcribe_single(wav_path)
                if result is not None:
                    # Lưu timestamps JSON
                    self._save_json(result)
                    results.append(result)
                    logger.info(f"[OK] {filename} - '{result.full_text[:50]}...'")
                else:
                    logger.warning(f"[SKIP] Không có speech: {filename}")
            except Exception as e:
                logger.error(f"[FAIL] Lỗi transcribe {filename}: {e}")

        logger.info(f"Hoàn thành! Đã transcribe {len(results)} files mới.")
        return results

    # def _transcribe_single(self, wav_path: str) -> Optional[TranscriptionResult]:
    #     """
    #     Transcribe 1 file WAV, trả về text + word-level timestamps.
        
    #     Sử dụng return_hypotheses=True để lấy thông tin timestep từ CTC.
    #     Sau đó nhóm characters thành words dựa trên khoảng trắng.
        
    #     Args:
    #         wav_path: Đường dẫn file WAV
            
    #     Returns:
    #         TranscriptionResult hoặc None nếu không có speech
    #     """
    #     import soundfile as sf

    #     # Lấy duration của file audio
    #     info = sf.info(wav_path)
    #     duration = info.duration

    #     # Transcribe với return_hypotheses=True để lấy timestep info
    #     hypotheses = self.model.transcribe(
    #         [wav_path],
    #         return_hypotheses=True,
    #         batch_size=1
    #     )

    #     # hypotheses là list of Hypothesis objects
    #     if not hypotheses or len(hypotheses) == 0:
    #         return None

    #     # Lấy hypothesis đầu tiên (chỉ có 1 file)
    #     hypothesis = hypotheses[0]

    #     # Lấy full text
    #     if hasattr(hypothesis, 'text'):
    #         full_text = hypothesis.text
    #     else:
    #         full_text = str(hypothesis)

    #     if not full_text or not full_text.strip():
    #         return None

    #     # Trích xuất word-level timestamps từ CTC alignment
    #     word_timestamps = self._extract_word_timestamps(hypothesis, duration)

    #     return TranscriptionResult(
    #         wav_path=wav_path,
    #         full_text=full_text,
    #         word_timestamps=word_timestamps,
    #         duration=duration
    #     )
    
    def _transcribe_single(
        self,
        wav_path: str
    ) -> Optional[TranscriptionResult]:

        import soundfile as sf
        import librosa
        import tempfile

        info = sf.info(wav_path)
        duration = info.duration

        # convert mono nếu cần
        if info.channels > 1:
            logger.info(
                f"Đang convert {os.path.basename(wav_path)} "
                f"từ {info.channels} kênh sang mono..."
            )

            data, samplerate = sf.read(wav_path)
            mono_data = data.mean(axis=1)

            sf.write(wav_path, mono_data, samplerate)

        # load toàn bộ audio
        audio, sr = librosa.load(wav_path, sr=16000)

        chunk_sec = 20
        overlap_sec = 5

        chunk_size = int(chunk_sec * sr)
        stride = int((chunk_sec - overlap_sec) * sr)

        all_texts = []
        all_word_timestamps = []

        for start in range(0, len(audio), stride):

            end = start + chunk_size

            chunk = audio[start:end]

            # bỏ chunk quá ngắn
            if len(chunk) < sr:
                continue

            # Tính offset thực tế dựa vào vị trí chunk trong audio
            current_offset = start / sr

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as tmp:

                sf.write(tmp.name, chunk, sr)

                try:

                    with torch.no_grad():
                        if "chunkformer" in self.model_name:
                            # WeNet transcribe takes a single wave file path and returns a result object
                            result = self.model.transcribe(tmp.name)
                            hypotheses = [result] if result else None
                        else:
                            hypotheses = self.model.transcribe(
                                [tmp.name],
                                return_hypotheses=True,
                                batch_size=1
                            )

                    if not hypotheses:
                        continue

                    hypothesis = hypotheses[0]

                    if hasattr(hypothesis, 'text'):
                        text = hypothesis.text
                    else:
                        text = str(hypothesis)

                    if text.strip():

                        all_texts.append(text)

                        word_timestamps = self._extract_word_timestamps(
                            hypothesis,
                            chunk_sec
                        )

                        # cộng offset thời gian
                        for w in word_timestamps:
                            w.start_time += current_offset
                            w.end_time += current_offset

                        all_word_timestamps.extend(word_timestamps)

                finally:
                    os.remove(tmp.name)

        if not all_texts:
            return None

        full_text = " ".join(all_texts)

        return TranscriptionResult(
            wav_path=wav_path,
            full_text=full_text,
            word_timestamps=all_word_timestamps,
            duration=duration
        )
    
    def _extract_word_timestamps(
        self, hypothesis, audio_duration: float
    ) -> List[WordTimestamp]:
        """
        Trích xuất word-level timestamps từ CTC hypothesis.
        
        CTC model output character-level alignments. 
        Nhóm characters thành words dựa trên khoảng trắng (space).
        
        Args:
            hypothesis: NeMo Hypothesis object chứa timestep info
            audio_duration: Tổng thời lượng audio (seconds)
            
        Returns:
            Danh sách WordTimestamp
        """
        # Lấy text từ hypothesis
        if hasattr(hypothesis, 'text'):
            text = hypothesis.text
        else:
            text = str(hypothesis)

        if not text or not text.strip():
            return []

        # Thử lấy timestep information từ hypothesis
        # NeMo CTC models có thể cung cấp timesteps qua nhiều cách
        timesteps = None

        # Cách 1: hypothesis.timestep (NeMo >= 1.20)
        if hasattr(hypothesis, 'timestep') and hypothesis.timestep is not None:
            timesteps = hypothesis.timestep

        # Cách 2: hypothesis.alignments
        if timesteps is None and hasattr(hypothesis, 'alignments') and hypothesis.alignments is not None:
            timesteps = hypothesis.alignments

        # Nếu có timestep info, dùng để tính word timestamps chính xác
        if timesteps is not None:
            return self._timestamps_from_ctc_timesteps(timesteps, text, audio_duration)

        # Fallback: chia đều timestamps theo số ký tự
        # (khi model không cung cấp timestep info)
        return self._timestamps_from_uniform_split(text, audio_duration)

    def _timestamps_from_ctc_timesteps(
        self, timesteps, text: str, audio_duration: float
    ) -> List[WordTimestamp]:
        """
        Tính word timestamps từ CTC timestep information.
        
        CTC timesteps cho biết frame index của mỗi character.
        Chuyển frame index -> seconds, rồi nhóm thành words.
        
        Args:
            timesteps: Timestep data từ hypothesis
            text: Full transcript text
            audio_duration: Tổng thời lượng audio
            
        Returns:
            Danh sách WordTimestamp
        """
        # Xác định số frames và thời gian mỗi frame
        # NeMo CTC models thường dùng stride 0.04s (40ms) hoặc 0.02s (20ms)
        # Tính từ tổng số frames và audio duration

        # Lấy character-level timestamps
        char_times = []

        if isinstance(timesteps, dict):
            # Format: {'timestep': [...], 'char': [...]}
            if 'timestep' in timesteps:
                frame_indices = timesteps['timestep']
                # Tính time_per_frame từ max frame index và audio duration
                max_frame = max(frame_indices) if frame_indices else 1
                time_per_frame = audio_duration / (max_frame + 1)
                char_times = [idx * time_per_frame for idx in frame_indices]
        elif isinstance(timesteps, (list, np.ndarray)):
            # Format: list of frame indices hoặc timestamps trực tiếp
            ts_list = list(timesteps) if isinstance(timesteps, np.ndarray) else timesteps

            if len(ts_list) > 0:
                # Kiểm tra xem đây là frame indices hay timestamps (seconds)
                max_val = max(ts_list) if ts_list else 0

                if max_val > audio_duration * 2:
                    # Đây là frame indices, cần convert sang seconds
                    time_per_frame = audio_duration / (max_val + 1)
                    char_times = [idx * time_per_frame for idx in ts_list]
                else:
                    # Đây đã là timestamps (seconds)
                    char_times = [float(t) for t in ts_list]

        # Nếu không parse được timesteps, fallback
        if not char_times:
            return self._timestamps_from_uniform_split(text, audio_duration)

        # Nhóm characters thành words dựa trên space
        # Mỗi character trong text tương ứng với 1 entry trong char_times
        words = []
        current_word_chars = []
        current_word_start_idx = 0

        text_chars = list(text)

        # Đảm bảo số char_times khớp với số characters (không tính spaces)
        non_space_count = sum(1 for c in text_chars if c != ' ')

        if len(char_times) < non_space_count:
            # Không đủ timestamps, fallback
            return self._timestamps_from_uniform_split(text, audio_duration)

        # Map char_times vào non-space characters
        time_idx = 0
        char_time_map = []  # (char, time) cho mỗi non-space char

        for char in text_chars:
            if char == ' ':
                char_time_map.append((char, None))
            else:
                if time_idx < len(char_times):
                    char_time_map.append((char, char_times[time_idx]))
                    time_idx += 1
                else:
                    char_time_map.append((char, None))

        # Nhóm thành words
        word_timestamps = []
        current_word = ""
        word_start_time = None
        word_end_time = None

        for char, time in char_time_map:
            if char == ' ':
                # Kết thúc word hiện tại
                if current_word and word_start_time is not None:
                    word_timestamps.append(WordTimestamp(
                        word=current_word,
                        start_time=round(word_start_time, 3),
                        end_time=round(word_end_time if word_end_time else word_start_time + 0.05, 3)
                    ))
                current_word = ""
                word_start_time = None
                word_end_time = None
            else:
                current_word += char
                if time is not None:
                    if word_start_time is None:
                        word_start_time = time
                    word_end_time = time

        # Thêm word cuối cùng
        if current_word and word_start_time is not None:
            # End time của word cuối = thời gian char cuối + khoảng nhỏ
            end_time = word_end_time if word_end_time else word_start_time
            # Đảm bảo end > start
            if end_time <= word_start_time:
                end_time = min(word_start_time + 0.05, audio_duration)
            word_timestamps.append(WordTimestamp(
                word=current_word,
                start_time=round(word_start_time, 3),
                end_time=round(end_time, 3)
            ))

        # Đảm bảo end_time > start_time cho mỗi word
        for i, wt in enumerate(word_timestamps):
            if wt.end_time <= wt.start_time:
                # Sử dụng start_time của word tiếp theo hoặc thêm offset nhỏ
                if i + 1 < len(word_timestamps):
                    word_timestamps[i] = WordTimestamp(
                        word=wt.word,
                        start_time=wt.start_time,
                        end_time=min(wt.start_time + 0.05, word_timestamps[i + 1].start_time)
                    )
                else:
                    word_timestamps[i] = WordTimestamp(
                        word=wt.word,
                        start_time=wt.start_time,
                        end_time=min(wt.start_time + 0.05, audio_duration)
                    )

        return word_timestamps

    def _timestamps_from_uniform_split(
        self, text: str, audio_duration: float
    ) -> List[WordTimestamp]:
        """
        Fallback: chia đều timestamps theo số ký tự của mỗi word.
        
        Khi model không cung cấp timestep info, ước lượng timestamps
        bằng cách phân bổ thời gian theo tỷ lệ số ký tự.
        
        Args:
            text: Full transcript text
            audio_duration: Tổng thời lượng audio
            
        Returns:
            Danh sách WordTimestamp (ước lượng)
        """
        words = text.strip().split()
        if not words:
            return []

        # Tính tổng số ký tự (không tính spaces)
        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            return []

        # Phân bổ thời gian theo tỷ lệ số ký tự
        # Thêm padding nhỏ ở đầu và cuối
        padding = min(0.1, audio_duration * 0.02)
        usable_duration = audio_duration - 2 * padding

        word_timestamps = []
        current_time = padding

        for word in words:
            # Thời lượng tỷ lệ với số ký tự
            word_duration = (len(word) / total_chars) * usable_duration
            start_time = current_time
            end_time = current_time + word_duration

            word_timestamps.append(WordTimestamp(
                word=word,
                start_time=round(start_time, 3),
                end_time=round(end_time, 3)
            ))

            current_time = end_time

        return word_timestamps

    def _save_json(self, result: TranscriptionResult):
        """
        Lưu TranscriptionResult dạng JSON (cùng tên file .json).
        
        Format JSON:
        {
            "wav_path": "path/to/file.wav",
            "full_text": "transcribed text",
            "duration": 123.45,
            "word_timestamps": [
                {"word": "xin", "start_time": 0.5, "end_time": 0.8},
                ...
            ]
        }
        """
        # Tạo path JSON: cùng tên file, đổi extension thành .json
        json_path = os.path.splitext(result.wav_path)[0] + ".json"

        # Chuẩn bị data
        data = {
            "wav_path": result.wav_path,
            "full_text": result.full_text,
            "duration": round(result.duration, 2),
            "word_timestamps": [
                {
                    "word": wt.word,
                    "start_time": wt.start_time,
                    "end_time": wt.end_time
                }
                for wt in result.word_timestamps
            ]
        }

        # Lưu file JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Đã lưu: {json_path}")
