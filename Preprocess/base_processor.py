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
    print("Downloading mythicinfinity/libritts (clean) ...")
    ds = load_dataset("mythicinfinity/libritts", "clean", cache_dir=dataset_dir)
    print("Download completed!")
    return ds


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
            ds = load_dataset(
                file_type,
                data_files=files,
                split="train"
            )
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
                print(f"Found {len(parquet_files)} parquet files")
                ds = load_dataset(
                    "parquet",
                    data_files=parquet_files,
                    split="train"
                )
            elif arrow_files:
                print(f"Found {len(arrow_files)} arrow files")
                ds = load_dataset(
                    "arrow",
                    data_files=arrow_files,
                    split="train"
                )
            else:
                print("No parquet or arrow files found")
                return
        stats_path = os.path.join(self.processed_dir, f"stats_{self.dataset_name}.json")
        processed_files = set()

        # Resume: kiểm tra file đã xử lý
        for f in os.listdir(self.wavs_dir):
            if f.endswith(".wav"):
                processed_files.add(f)
        print(f"Found {len(processed_files)} processed files")

        # Load old stats if exists
        stats = {
            "total_seen": 0,
            "processed": 0,
            "duration_filtered": 0,
            "empty_text": 0,
            "errors": 0,
            "skipped_existing": 0
        }
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        stats = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                print(f"Warning: {stats_path} is empty or corrupted. Re-initializing stats.")

        stats["total_seen"] = 0
        stats["errors"] = 0
        stats["skipped_existing"] = 0
        stats["duration_filtered"] = 0
        stats["empty_text"] = 0

        metadata_path = os.path.join(self.output_dir, "metadata.csv")

        for idx, item in enumerate(tqdm(ds)):
            stats["total_seen"] += 1
            filename = f"{self.dataset_name}_{idx:06d}.wav"

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
                    # Fallback cho LibriTTS (HuggingFace arrow files dùng text_normalized hoặc text_original)
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
            except Exception as e:
                stats["errors"] += 1
                print(e)

            # Save stats realtime
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)

        print("Processing finished")
        print(json.dumps(stats, indent=4, ensure_ascii=False))




download_libritts()



processor = DatasetProcessor(
    raw_data_dir=os.path.join(target_dir, "libritts"),
    processed_dir=os.path.join(BASE_DIR, "Processed_DATA"),
    dataset_name = "libritts"
)
processor.process() 