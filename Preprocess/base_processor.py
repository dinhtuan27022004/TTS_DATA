import os
import glob
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from abc import ABC, abstractmethod
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import snapshot_download
HF_TOKEN = "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"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_dir = os.path.join(BASE_DIR, "data")

os.makedirs(target_dir, exist_ok=True)
def download_phoaudiobook():
    dataset_dir = os.path.join(target_dir, "PhoAudioBook")
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Downloading thivux/phoaudiobook to {dataset_dir}/ ...")

    snapshot_download(
        repo_id="thivux/phoaudiobook",
        repo_type="dataset",
        local_dir=dataset_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN
    )
    print("Download completed! All files are in", dataset_dir)

def download_vivoice():
    dataset_dir = os.path.join(target_dir, "viVoice")
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Downloading capleaf/viVoice to {dataset_dir}/ ...")

    snapshot_download(
        repo_id="capleaf/viVoice",
        repo_type="dataset",
        local_dir=dataset_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN
    )
    print("Download completed! All files are in", dataset_dir)

def download_vietnam_celeb():
    dataset_dir = os.path.join(target_dir, "Vietnam-Celeb")
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Downloading hustep-lab/Vietnam-Celeb to {dataset_dir}/ ...")

    snapshot_download(
        repo_id="hustep-lab/Vietnam-Celeb",
        repo_type="dataset",
        local_dir=dataset_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN
    )
    print("Download completed! All files are in", dataset_dir)


def download_libritts():
    dataset_dir = os.path.join(target_dir, "libritts")
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Downloading mythicinfinity/libritts to {dataset_dir}/ ...")

    snapshot_download(
        repo_id="mythicinfinity/libritts",
        repo_type="dataset",
        local_dir=dataset_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN
    )
    print("Download completed! All files are in", dataset_dir)


class DatasetProcessor:
    def __init__(
        self,
        dataset_name,
        raw_data_dir,
        processed_dir,
        text_column="text",
        target_sr=24000,
        output_format="pairs",  # "pairs" = wav+txt, "metadata" = metadata.csv
        file_pattern=None,      # e.g., "*train.clean.100*.arrow"
    ):
        self.dataset_name = dataset_name
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        self.text_column = text_column
        self.target_sr = target_sr
        self.min_duration = 2.0
        self.max_duration = 30.0
        self.output_format = output_format
        self.output_dir = os.path.join(processed_dir, dataset_name)
        self.file_pattern = file_pattern

        if output_format == "metadata":
            self.wavs_dir = os.path.join(self.output_dir, "wavs")
        else:
            self.wavs_dir = self.output_dir

        os.makedirs(self.wavs_dir, exist_ok=True)

    def save_audio(self, audio_array, source_sr, filename):
        if source_sr != self.target_sr:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=source_sr,
                target_sr=self.target_sr
            )
        output_path = os.path.join(self.wavs_dir, filename)
        sf.write(output_path, audio_array, self.target_sr)
        duration = len(audio_array) / self.target_sr
        return duration

    def process(self):
        if self.file_pattern:
            files = glob.glob(
                os.path.join(self.raw_data_dir, "**", self.file_pattern),
                recursive=True
            )
            if not files:
                print(f"No files found matching pattern: {self.file_pattern}")
                return
            
            first_file = files[0]
            if first_file.endswith(".parquet"):
                file_type = "parquet"
            elif first_file.endswith(".arrow"):
                file_type = "arrow"
            else:
                raise ValueError(f"Unsupported file format: {first_file}")
                
            print(f"Found {len(files)} files matching pattern '{self.file_pattern}' (Type: {file_type})")
        else:
            parquet_files = glob.glob(
                os.path.join(self.raw_data_dir, "**", "*.parquet"),
                recursive=True
            )
            arrow_files = glob.glob(
                os.path.join(self.raw_data_dir, "**", "*.arrow"),
                recursive=True
            )

            if parquet_files:
                files = parquet_files
                file_type = "parquet"
            elif arrow_files:
                files = arrow_files
                file_type = "arrow"
            else:
                print("No parquet or arrow files found")
                return
            print(f"Found {len(files)} {file_type} files")

        files.sort() # Ensure consistent order

        stats_path = os.path.join(self.processed_dir, f"stats_{self.dataset_name}.json")
        processed_files = set()

        # Resume: kiểm tra file đã xử lý từ disk (source of truth)
        for f in os.listdir(self.wavs_dir):
            if f.endswith(".wav"):
                processed_files.add(f)
        print(f"Found {len(processed_files)} already processed files on disk")

        # Stats: reset session counters mỗi lần chạy
        # 'processed' lấy từ số file thực trên disk, không tin vào JSON cũ
        stats = {
            "total_seen": 0,
            "processed": len(processed_files),  # Số file thực tế trên disk
            "duration_filtered": 0,
            "empty_text": 0,
            "errors": 0,
            "skipped_existing": len(processed_files)  # Ước tính ban đầu
        }
        print(f"Stats initialized: processed={stats['processed']} files on disk")

        metadata_path = os.path.join(self.output_dir, "metadata.csv")

        # Process each file sequentially
        for file_path in files:
            print(f"\n--- Processing file: {file_path} ---")
            try:
                ds = load_dataset(file_type, data_files=[file_path], split="train")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
            # Dùng full stem để tránh collision giữa các shard:
            # train.clean.360-00062-of-00069.parquet → train_clean_360-00062-of-00069
            stem = os.path.splitext(os.path.basename(file_path))[0]  # bỏ .parquet/.arrow
            base_filename = stem.replace('.', '_')

            for local_idx, item in enumerate(tqdm(ds, desc=f"Extracting {base_filename}")):
                stats["total_seen"] += 1
                # Format: libritts_train-00000-of-00500_000001.wav
                filename = f"{self.dataset_name}_{base_filename}_{local_idx:06d}.wav"

                # Resume support
                if filename in processed_files:
                    stats["skipped_existing"] += 1
                    continue

                try:
                    audio = item["audio"]
                    audio_array = np.asarray(
                        audio["array"],
                        dtype=np.float32
                    )
                    sr = audio["sampling_rate"]
                    duration = len(audio_array) / sr

                    # Duration filter
                    if not (self.min_duration <= duration <= self.max_duration):
                        stats["duration_filtered"] += 1
                        continue

                    text = item.get(self.text_column)
                    if text is None:
                        # Fallback cho LibriTTS
                        text = item.get("text_normalized") or item.get("text_original")
                    text = str(text or "").strip()
                    # Empty text
                    if not text:
                        stats["empty_text"] += 1
                        continue

                    final_duration = self.save_audio(audio_array, sr, filename)

                    if self.output_format == "pairs":
                        # Lưu file .txt cùng tên với .wav
                        txt_filename = filename.replace(".wav", ".txt")
                        txt_path = os.path.join(self.wavs_dir, txt_filename)
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(text)
                    else:
                        # Append metadata.csv
                        rel_path = f"wavs/{filename}"
                        with open(metadata_path, "a", encoding="utf-8") as f:
                            f.write(f"{rel_path}|{text}|{round(final_duration, 3)}\n")

                    stats["processed"] += 1
                    processed_files.add(filename)
                except Exception as e:
                    stats["errors"] += 1
                    print(e)

                # Save stats mỗi 500 item để giảm I/O
                if stats["total_seen"] % 500 == 0:
                    with open(stats_path, "w", encoding="utf-8") as f:
                        json.dump(stats, f, indent=4, ensure_ascii=False)
            
            # Clean up cache and delete the original file
            try:
                ds.cleanup_cache_files()
            except Exception as e:
                print(f"Could not clean cache: {e}")

            try:
                os.remove(file_path)
                print(f"-> Deleted original file: {file_path}")
            except Exception as e:
                print(f"-> Could not delete {file_path}: {e}")

        print("Processing finished")
        print(json.dumps(stats, indent=4, ensure_ascii=False))


# Re-download vì toàn bộ parquet đã bị xóa
download_libritts()


processor = DatasetProcessor(
    raw_data_dir=os.path.join(target_dir, "libritts"),
    processed_dir=os.path.join(BASE_DIR, "Processed_DATA"),
    dataset_name="libritts",
    text_column="text_normalized",
)
processor.process()