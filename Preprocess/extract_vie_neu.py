#!/usr/bin/env python3
"""
Script to extract audio files and metadata from VieNeu-TTS arrow files.
"""

import os
# Force offline mode for Hugging Face datasets to prevent hanging on network calls
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import glob
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio and metadata from VieNeu-TTS arrow dataset.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/home/reg/TTS_DATA/VieNeu-TTS/pnnbao-ump___vie_neu-tts/default/0.0.0/a5f8845053018f68467d45d5804b83711c7c1a01",
        help="Path to the directory containing VieNeu-TTS arrow shards."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/reg/TTS_DATA/extracted_vie_neu",
        help="Path to save the extracted dataset."
    )
    parser.add_argument(
        "--flat", 
        action="store_true",
        help="If set, save all WAV files in a single flat directory. Otherwise, group by speaker."
    )
    parser.add_argument(
        "--write_lab", 
        action="store_true",
        help="If set, write individual .lab transcript files next to each .wav file."
    )
    parser.add_argument(
        "--write_txt", 
        action="store_true",
        help="If set, write individual .txt transcript files next to each .wav file."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Gather all arrow shards
    arrow_files = sorted(glob.glob(os.path.join(args.input_dir, "*.arrow")))
    if not arrow_files:
        print(f"Error: No .arrow files found in {args.input_dir}")
        return
        
    print(f"Found {len(arrow_files)} arrow shards in {args.input_dir}")
    print(f"Loading dataset offline...")
    
    # Load dataset
    dataset = load_dataset("arrow", data_files=arrow_files, split="train")
    total_examples = len(dataset)
    print(f"Loaded {total_examples} examples successfully.")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting extraction to {args.output_dir}...")
    
    metadata = []
    errors = 0
    
    for example in tqdm(dataset, desc="Extracting"):
        try:
            example_id = example.get("_id")
            speaker = example.get("speaker", "unknown")
            audio_info = example.get("audio")
            
            if not example_id or not audio_info:
                continue
                
            audio_bytes = audio_info.get("bytes")
            if not audio_bytes:
                continue
                
            # Determine paths
            if args.flat:
                wav_rel_path = f"wavs/{example_id}.wav"
            else:
                wav_rel_path = f"wavs/{speaker}/{example_id}.wav"
                
            wav_abs_path = os.path.join(args.output_dir, wav_rel_path)
            os.makedirs(os.path.dirname(wav_abs_path), exist_ok=True)
            
            # Save WAV file directly from bytes (no decoding/encoding overhead)
            with open(wav_abs_path, "wb") as f:
                f.write(audio_bytes)
                
            # Write .lab/txt file if requested
            text = example.get("text", "")
            if args.write_lab:
                lab_abs_path = os.path.splitext(wav_abs_path)[0] + ".lab"
                with open(lab_abs_path, "w", encoding="utf-8") as f:
                    f.write(text)
            if args.write_txt:
                txt_abs_path = os.path.splitext(wav_abs_path)[0] + ".txt"
                with open(txt_abs_path, "w", encoding="utf-8") as f:
                    f.write(text)
                    
            # Record metadata
            metadata.append({
                "id": example_id,
                "audio_path": wav_rel_path,
                "text": text,
                "phonemized_text": example.get("phonemized_text", ""),
                "duration": example.get("duration", 0.0),
                "speaker": speaker,
                "gender": example.get("gender", "unknown"),
                "language": example.get("language", "vi")
            })
        except Exception as e:
            errors += 1
            
    print(f"Extraction completed. Extracted: {len(metadata)}, Errors: {errors}")
    
    # Save metadata to CSV
    if metadata:
        df = pd.DataFrame(metadata)
        # Reorder columns
        cols = ["id", "audio_path", "text", "phonemized_text", "duration", "speaker", "gender", "language"]
        df = df[cols]
        csv_path = os.path.join(args.output_dir, "metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata CSV to {csv_path}")

if __name__ == "__main__":
    main()
