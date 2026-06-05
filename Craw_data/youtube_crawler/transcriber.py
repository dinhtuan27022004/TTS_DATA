"""
Phase 4: Transcription
Transcribe audio tiếng Việt bằng nvidia/parakeet-ctc-0.6b-vi.
Trích xuất word-level timestamps từ CTC model output.
"""

import os
import json
import logging
from typing import List, Optional, Tuple

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
        model_name: str = "nvidia/parakeet-ctc-0.6b-vi",
        use_punctuation: bool = False,
        pause_threshold: float = 0.05
    ):
        """
        Args:
            input_dir: Thư mục Step_1 chứa vocals WAV
            model_name: Tên model NeMo ASR
            use_punctuation: Bật model phục hồi dấu câu
            pause_threshold: Ngưỡng thời gian (giây) để giữ lại dấu câu
        """
        self.input_dir = input_dir
        self.model_name = model_name
        self.use_punctuation = use_punctuation
        self.pause_threshold = pause_threshold

        # Model sẽ được load khi cần (lazy loading)
        self.model = None
        self.punct_model = None

    def _load_model(self):
        """
        Load model bằng NeMo toolkit hoặc ChunkFormer.
        Tự động chọn GPU nếu có.
        """
        if torch.cuda.is_available():
            try:
                # Thiết lập giới hạn VRAM 5GB
                total_memory = torch.cuda.get_device_properties(0).total_memory
                target_memory = 5 * 1024**3  # 5 GB
                if target_memory < total_memory:
                    fraction = target_memory / total_memory
                    torch.cuda.set_per_process_memory_fraction(fraction, 0)
                    logger.info(f"Đã giới hạn VRAM ở mức 5GB ({(fraction*100):.1f}% của GPU)")
            except Exception as e:
                logger.warning(f"Không thể thiết lập giới hạn VRAM: {e}")

        if "chunkformer" in self.model_name.lower():
            from chunkformer import ChunkFormerModel
            logger.info(f"Đang load ChunkFormer model từ: {self.model_name}...")
            self.model = ChunkFormerModel.from_pretrained(self.model_name)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Sử dụng GPU (CUDA) cho ChunkFormer")
            logger.info(f"ChunkFormer Model {self.model_name} đã load thành công!")
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

        if self.use_punctuation:
            from transformers import pipeline
            logger.info("Đang load Punctuation Model bmd1905/vietnamese-correction-v2...")
            self.punct_model = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2", device=0 if torch.cuda.is_available() else -1)
            logger.info("Punctuation Model đã load thành công!")


    def transcribe_all(self) -> List[TranscriptionResult]:
        import time
        results = []
        max_empty_retries = 3
        empty_retries = 0
        processed_demucs_path = os.path.join(self.input_dir, "processed_demucs.json")

        # Load model trước vòng lặp nếu chưa load
        if self.model is None:
            self._load_model()

        while True:
            # Quét trực tiếp thư mục Step_1 để không bỏ sót bất kỳ file nào (kể cả file mồ côi)
            demucs_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".wav")]
            demucs_files.sort()

            # Kiểm tra file nào đã có .json (resume)
            files_to_process = []
            for filename in demucs_files:
                json_path = os.path.join(
                    self.input_dir,
                    os.path.splitext(filename)[0] + ".json"
                )
                if os.path.exists(json_path):
                    continue
                
                # Đảm bảo file wav thực sự tồn tại
                wav_path = os.path.join(self.input_dir, filename)
                if os.path.exists(wav_path):
                    files_to_process.append(filename)

            if not files_to_process:
                empty_retries += 1
                if empty_retries > max_empty_retries:
                    logger.info("Không có file mới nào sau nhiều lần thử. Kết thúc Phase 4.")
                    break
                logger.info(f"Chưa có file mới. Chờ 60 giây và thử lại... ({empty_retries}/{max_empty_retries})")
                time.sleep(60)
                continue

            # Reset retries vì tìm thấy file mới
            empty_retries = 0
            logger.info(f"Cần transcribe thêm lô mới: {len(files_to_process)} files")

            # Transcribe từng file
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
                        # Lưu một file json trống để đánh dấu là đã xử lý nhưng không có speech
                        empty_result = TranscriptionResult(wav_path=wav_path, full_text="", word_timestamps=[], duration=0.0)
                        self._save_json(empty_result)
                        logger.warning(f"[SKIP] Không có speech: {filename}")
                    
                    # Giải phóng VRAM sau mỗi file để tránh dồn bộ nhớ
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error(f"[CRITICAL] Hết VRAM (OOM) khi xử lý {filename}. Chương trình sẽ dừng lại!")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import sys
                        sys.exit(1)
                    else:
                        logger.error(f"[FAIL] Lỗi transcribe {filename}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"[FAIL] Lỗi transcribe {filename}: {e}", exc_info=True)

        logger.info(f"Hoàn thành! Đã transcribe tổng cộng {len(results)} files mới trong phiên này.")
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

        if "chunkformer" in self.model_name.lower():
            logger.info(f"Sử dụng endless_decode cho file: {os.path.basename(wav_path)}")
            results = self.model.endless_decode(
                audio_path=wav_path,
                chunk_size=64,
                left_context_size=128,
                right_context_size=128,
                total_batch_duration=60,  # Giảm từ 360 xuống 60 để tiết kiệm VRAM
                return_timestamps=True
            )
            
            all_word_timestamps = []
            full_text_parts = []
            
            for item in results:
                text = item.get('decode', '')
                if not text.strip(): continue
                
                full_text_parts.append(text)
                
                start_str = item.get('start', '00:00:00:000')
                end_str = item.get('end', '00:00:00:000')
                
                def time_str_to_sec(t_str):
                    parts = t_str.split(':')
                    if len(parts) == 4:
                        h, m, s, ms = parts
                        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
                    return 0.0
                    
                start_sec = time_str_to_sec(start_str)
                end_sec = time_str_to_sec(end_str)
                
                words = text.strip().split()
                if not words: continue
                
                total_chars = sum(len(w) for w in words)
                if total_chars == 0: continue
                
                seg_duration = end_sec - start_sec
                current_time = start_sec
                
                for w in words:
                    w_duration = (len(w) / total_chars) * seg_duration
                    all_word_timestamps.append(WordTimestamp(
                        word=w,
                        start_time=round(current_time, 3),
                        end_time=round(current_time + w_duration, 3)
                    ))
                    current_time += w_duration
            
            if not full_text_parts:
                return None
                
            full_text = " ".join(full_text_parts)
            
            if self.use_punctuation:
                full_text, all_word_timestamps = self._restore_punctuation_and_verify(full_text, all_word_timestamps)
                
            return TranscriptionResult(
                wav_path=wav_path,
                full_text=full_text,
                word_timestamps=all_word_timestamps,
                duration=duration
            )

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
        
        if self.use_punctuation:
            full_text, all_word_timestamps = self._restore_punctuation_and_verify(full_text, all_word_timestamps)

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

    def _restore_punctuation_and_verify(
        self,
        text: str,
        word_timestamps: List[WordTimestamp]
    ) -> Tuple[str, List[WordTimestamp]]:
        """
        Phục hồi dấu câu và viết hoa, xác minh dựa trên timestamps.
        Đảm bảo không bị viết hoa ngẫu nhiên (chỉ viết hoa đầu câu).
        """
        if not self.punct_model or not text.strip() or not word_timestamps:
            return text, word_timestamps
            
        logger.info("Đang phục hồi dấu câu và căn chỉnh...")
        import re
        
        # Cắt nhỏ text để tránh vượt quá max_length của model
        words = text.split()
        chunk_size = 50
        punctuated_chunks = []
        
        try:
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                res = self.punct_model(chunk, max_new_tokens=256)
                if isinstance(res, list) and len(res) > 0 and 'generated_text' in res[0]:
                    punctuated_chunks.append(res[0]['generated_text'])
                else:
                    punctuated_chunks.append(chunk)
        except Exception as e:
            logger.error(f"Lỗi khi chạy punctuation model: {e}")
            return text, word_timestamps
            
        punctuated_text = " ".join(punctuated_chunks)
        
        # Alignment
        orig_words = [wt.word for wt in word_timestamps]
        punct_words = punctuated_text.split()
        
        def clean_word(w):
            return re.sub(r'[^\w\s]', '', w).lower()
            
        orig_idx = 0
        punct_idx = 0
        verified_word_timestamps = []
        
        # Ngưỡng (s) để giữ lại dấu câu
        # Dấu phẩy cần gap ít hơn dấu chấm
        comma_threshold = 0.08
        period_threshold = 0.15
        
        while orig_idx < len(orig_words) and punct_idx < len(punct_words):
            o_word = orig_words[orig_idx]
            p_word = punct_words[punct_idx]
            
            c_o = clean_word(o_word)
            c_p = clean_word(p_word)
            
            # Khớp nếu từ gốc và từ LLM có chứa nhau (sau khi xóa dấu)
            if c_o and c_p and (c_o == c_p or c_o in c_p or c_p in c_o):
                wt = word_timestamps[orig_idx]
                # Lưu ý: LLM có thể trả về viết hoa ngẫu nhiên, ta sẽ fix ở bước sau
                
                has_punct = bool(re.search(r'[.,!?]$', p_word))
                
                if has_punct:
                    current_end = wt.end_time
                    next_start = word_timestamps[orig_idx+1].start_time if orig_idx + 1 < len(word_timestamps) else current_end + 1.0
                    gap = next_start - current_end
                    
                    is_comma = bool(re.search(r'[,]$', p_word))
                    threshold = comma_threshold if is_comma else period_threshold
                    
                    if gap < threshold:
                        # Khoảng lặng quá ngắn, bỏ dấu câu
                        p_word = re.sub(r'[.,!?]+$', '', p_word)
                        
                verified_word_timestamps.append(WordTimestamp(
                    word=p_word,
                    start_time=wt.start_time,
                    end_time=wt.end_time
                ))
                orig_idx += 1
                punct_idx += 1
            else:
                if orig_idx + 1 < len(orig_words) and clean_word(orig_words[orig_idx+1]) == c_p:
                    verified_word_timestamps.append(word_timestamps[orig_idx])
                    orig_idx += 1
                elif punct_idx + 1 < len(punct_words) and clean_word(punct_words[punct_idx+1]) == c_o:
                    punct_idx += 1
                else:
                    verified_word_timestamps.append(word_timestamps[orig_idx])
                    orig_idx += 1
                    punct_idx += 1
                    
        # Append remaining words
        while orig_idx < len(word_timestamps):
            verified_word_timestamps.append(word_timestamps[orig_idx])
            orig_idx += 1
            
        # Post-process Casing (Chuẩn hóa viết hoa)
        # Bắt buộc chữ thường, chỉ viết hoa từ đầu tiên hoặc từ sau dấu chấm/hỏi/chấm than
        capitalize_next = True
        for i in range(len(verified_word_timestamps)):
            w = verified_word_timestamps[i].word
            
            # Tách phần dấu câu ở cuối (nếu có) để xét viết hoa phần chữ
            match = re.search(r'^(.+?)([.,!?]*)$', w)
            if match:
                core_word = match.group(1)
                punct = match.group(2)
                
                if capitalize_next:
                    core_word = core_word.capitalize()
                    capitalize_next = False
                else:
                    core_word = core_word.lower()
                    
                w = core_word + punct
                
                # Nếu từ này kết thúc bằng dấu chấm/hỏi/chấm than, từ tiếp theo viết hoa
                if re.search(r'[.!?]$', punct):
                    capitalize_next = True
                    
            verified_word_timestamps[i].word = w
            
        new_full_text = " ".join([wt.word for wt in verified_word_timestamps])
        return new_full_text, verified_word_timestamps

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
