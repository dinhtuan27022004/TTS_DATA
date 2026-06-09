import os
import glob
import json
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Configuration
HF_TOKEN = "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"
BASE_DIR = "/home/reg/TTS_DATA"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(RAW_DATA_DIR, "vn_multi_dialect")
TARGET_SR = 24000
MIN_DURATION = 2.0
MAX_DURATION = 30.0

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
for d in ["bắc", "trung", "nam"]:
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

# Dialect mapping from province/region strings
PROVINCE_DIALECT_MAP = {
    # Northern (Bắc)
    "hanoi": "bắc", "ha noi": "bắc", "bac": "bắc", "north": "bắc", "haiphong": "bắc", "hai phong": "bắc",
    "quangninh": "bắc", "quang ninh": "bắc", "thaibinh": "bắc", "thai binh": "bắc", "namdinh": "bắc",
    "nam dinh": "bắc", "ninhbinh": "bắc", "ninh binh": "bắc", "langson": "bắc", "lang son": "bắc",
    # Central (Trung)
    "hue": "trung", "thua thien hue": "trung", "danang": "trung", "da nang": "trung", "quangnam": "trung",
    "quang nam": "trung", "nghean": "trung", "nghe an": "trung", "hatinh": "trung", "ha tinh": "trung",
    "quangbinh": "trung", "quang binh": "trung", "central": "trung", "trung": "trung",
    # Southern (Nam)
    "hcm": "nam", "ho chi minh": "nam", "saigon": "nam", "sai gon": "nam", "cantho": "nam", "can tho": "nam",
    "binhduong": "nam", "binh duong": "nam", "dongnai": "nam", "dong nai": "nam", "longan": "nam",
    "long an": "nam", "south": "nam", "nam": "nam"
}

def get_dialect_group(dialect_str):
    if not dialect_str:
        return "bắc"  # Default fallback
    
    dialect_lower = str(dialect_str).lower().strip()
    
    # Try direct mapping
    if dialect_lower in PROVINCE_DIALECT_MAP:
        return PROVINCE_DIALECT_MAP[dialect_lower]
        
    # Search for keywords
    for key, value in PROVINCE_DIALECT_MAP.items():
        if key in dialect_lower:
            return value
            
    return "bắc"  # Fallback

def download_datasets():
    print("=== Downloading Datasets ===")
    datasets_to_download = [
        {"repo_id": "nguyendv02/ViMD_Dataset", "dir": "ViMD_Dataset", "gated": False},
        {"repo_id": "capleaf/viVoice", "dir": "viVoice", "gated": True},
        {"repo_id": "NhutP/VietSpeech", "dir": "VietSpeech", "gated": False}
    ]
    
    for ds in datasets_to_download:
        local_path = os.path.join(RAW_DATA_DIR, ds["dir"])
        if os.path.exists(local_path) and os.listdir(local_path):
            print(f"Dataset {ds['repo_id']} already exists at {local_path}. Skipping download.")
            continue
            
        print(f"Downloading {ds['repo_id']} to {local_path}...")
        try:
            snapshot_download(
                repo_id=ds["repo_id"],
                repo_type="dataset",
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=HF_TOKEN if ds["gated"] else None
            )
            print(f"Downloaded {ds['repo_id']} successfully.")
        except Exception as e:
            print(f"Failed to download {ds['repo_id']}: {e}")
            print("Please ensure internet connectivity is available or place the dataset manually.")

def process_dataset(dataset_name, dataset_dir, text_col="text", dialect_col="dialect"):
    print(f"\n=== Processing Dataset: {dataset_name} ===")
    parquet_files = glob.glob(os.path.join(dataset_dir, "**", "*.parquet"), recursive=True)
    arrow_files = glob.glob(os.path.join(dataset_dir, "**", "*.arrow"), recursive=True)
    files = parquet_files + arrow_files
    
    if not files:
        print(f"No parquet or arrow files found in {dataset_dir}")
        return []
        
    print(f"Found {len(files)} files to process.")
    files.sort()
    
    metadata_records = []
    
    for file_path in files:
        print(f"Loading {os.path.basename(file_path)}...")
        file_type = "parquet" if file_path.endswith(".parquet") else "arrow"
        try:
            ds = load_dataset(file_type, data_files=[file_path], split="train")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
            
        stem = os.path.splitext(os.path.basename(file_path))[0].replace('.', '_')
        
        for idx, item in enumerate(tqdm(ds, desc=f"Extracting {stem}")):
            try:
                audio = item.get("audio")
                if audio is None:
                    continue
                    
                audio_array = np.asarray(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                duration = len(audio_array) / sr
                
                if not (MIN_DURATION <= duration <= MAX_DURATION):
                    continue
                    
                # Get text
                text = item.get(text_col) or item.get("text_normalized") or item.get("text_original")
                text = str(text or "").strip()
                if not text:
                    continue
                    
                # Determine dialect
                raw_dialect = item.get(dialect_col) or "bắc"
                dialect_group = get_dialect_group(raw_dialect)
                
                # Save normalized WAV
                filename = f"{dataset_name}_{stem}_{idx:06d}.wav"
                rel_wav_path = os.path.join(dialect_group, filename)
                abs_wav_path = os.path.join(OUTPUT_DIR, rel_wav_path)
                
                # Resample and write
                if sr != TARGET_SR:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
                sf.write(abs_wav_path, audio_array, TARGET_SR)
                
                # Write lab/txt file
                lab_path = abs_wav_path.replace(".wav", ".lab")
                with open(lab_path, "w", encoding="utf-8") as lf:
                    lf.write(text)
                    
                # Prepend dialect tag to transcript in metadata
                tagged_text = f"<{dialect_group}> {text}"
                metadata_records.append({
                    "wav_path": rel_wav_path,
                    "transcript": tagged_text,
                    "dialect": dialect_group
                })
                
            except Exception as e:
                # Silently skip individual errors during loop to keep tqdm output clean
                pass
                
        # Clean cache after processing file
        try:
            ds.cleanup_cache_files()
        except Exception:
            pass
            
    return metadata_records

def main():
    # Attempt download
    download_datasets()
    
    all_metadata = []
    
    # Process ViMD
    vimd_dir = os.path.join(RAW_DATA_DIR, "ViMD_Dataset")
    if os.path.exists(vimd_dir) and os.listdir(vimd_dir):
        vimd_meta = process_dataset("ViMD", vimd_dir, text_col="transcript", dialect_col="dialect")
        all_metadata.extend(vimd_meta)
        
    # Process viVoice
    vivoice_dir = os.path.join(RAW_DATA_DIR, "viVoice")
    if os.path.exists(vivoice_dir) and os.listdir(vivoice_dir):
        vivoice_meta = process_dataset("viVoice", vivoice_dir, text_col="text", dialect_col="dialect")
        all_metadata.extend(vivoice_meta)
        
    # Process VietSpeech
    vietspeech_dir = os.path.join(RAW_DATA_DIR, "VietSpeech")
    if os.path.exists(vietspeech_dir) and os.listdir(vietspeech_dir):
        vietspeech_meta = process_dataset("VietSpeech", vietspeech_dir, text_col="text", dialect_col="dialect")
        all_metadata.extend(vietspeech_meta)
        
    if all_metadata:
        # Save metadata.csv
        meta_df = pd.DataFrame(all_metadata)
        meta_path = os.path.join(OUTPUT_DIR, "metadata.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"\nSaved metadata CSV to {meta_path} with {len(meta_df)} records.")
        print(meta_df["dialect"].value_counts())
    else:
        print("\nNo records were processed. Please check if datasets exist under data/ directory.")

if __name__ == "__main__":
    main()
