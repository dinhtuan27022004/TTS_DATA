#!/usr/bin/env python3
"""
Quick script to generate individual .txt transcript files next to already extracted .wav files
using the metadata.csv file.
"""

import os
import pandas as pd
from tqdm import tqdm

def main():
    metadata_path = "/home/reg/TTS_DATA/extracted_vie_neu/metadata.csv"
    output_dir = "/home/reg/TTS_DATA/extracted_vie_neu"
    
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}")
        return
        
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    print(f"Found {len(df)} records. Writing .txt files...")
    
    success_count = 0
    error_count = 0
    
    for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Writing TXT files")):
        try:
            audio_path = row.audio_path
            text = row.text
            
            if pd.isna(audio_path) or pd.isna(text):
                continue
                
            # Construct the path to the corresponding .txt file
            # e.g., output_dir + / + wavs/speaker/id.wav -> output_dir + / + wavs/speaker/id.txt
            wav_abs_path = os.path.join(output_dir, audio_path)
            txt_abs_path = os.path.splitext(wav_abs_path)[0] + ".txt"
            
            # Ensure the directory exists (it should, but just in case)
            os.makedirs(os.path.dirname(txt_abs_path), exist_ok=True)
            
            with open(txt_abs_path, "w", encoding="utf-8") as f:
                f.write(str(text).strip())
                
            success_count += 1
        except Exception as e:
            error_count += 1
            
    print(f"Completed! Generated {success_count} .txt files. Errors: {error_count}")

if __name__ == "__main__":
    main()
