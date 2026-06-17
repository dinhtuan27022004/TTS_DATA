#!/usr/bin/env python3
"""
Script to calculate detailed statistics (total duration, average, speaker breakdown) 
for a directory of WAV files using python's built-in wave module.
"""

import os
import glob
import wave
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate statistics for a directory of WAV files.")
    parser.add_argument(
        "dir_path", 
        type=str, 
        nargs="?", 
        default="/home/reg/TTS_DATA/data/vie_neu",
        help="Path to the directory containing WAV files."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=16,
        help="Number of threads to use for scanning."
    )
    return parser.parse_args()

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return file_path, duration
    except Exception as e:
        return file_path, None

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"

def main():
    args = parse_args()
    
    if not os.path.exists(args.dir_path):
        print(f"Error: Directory '{args.dir_path}' does not exist.")
        return
        
    print(f"Scanning directory: {args.dir_path}")
    wav_files = glob.glob(os.path.join(args.dir_path, "**", "*.wav"), recursive=True)
    total_files = len(wav_files)
    
    if total_files == 0:
        print("No WAV files found.")
        return
        
    print(f"Found {total_files} WAV files. Reading headers...")
    
    durations = []
    speaker_durations = {}
    errors = 0
    
    # Use ThreadPoolExecutor to read headers concurrently
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(get_wav_duration, f) for f in wav_files]
        
        for future in tqdm(as_completed(futures), total=total_files, desc="Processing WAVs"):
            file_path, duration = future.result()
            if duration is not None:
                durations.append(duration)
                
                # Determine speaker from the parent directory name
                parent_dir = os.path.basename(os.path.dirname(file_path))
                # If parent dir is not the main dir, treat it as speaker_id
                if parent_dir and os.path.dirname(file_path) != os.path.abspath(args.dir_path):
                    speaker_id = parent_dir
                else:
                    speaker_id = "unknown"
                    
                speaker_durations[speaker_id] = speaker_durations.get(speaker_id, 0.0) + duration
            else:
                errors += 1
                
    if not durations:
        print("Failed to read duration from any WAV file.")
        return
        
    durations = np.array(durations)
    total_seconds = np.sum(durations)
    mean_duration = np.mean(durations)
    min_duration = np.min(durations)
    max_duration = np.max(durations)
    std_duration = np.std(durations)
    
    print("\n" + "="*50)
    print("                WAV FILE STATISTICS")
    print("="*50)
    print(f"Total WAV files:         {total_files:,}")
    print(f"Successfully processed:  {len(durations):,}")
    print(f"Errors / Failed:         {errors:,}")
    print("-"*50)
    print(f"Total Duration:          {format_time(total_seconds)} ({total_seconds:.2f} seconds)")
    print(f"Average Duration:        {mean_duration:.2f} seconds")
    print(f"Min Duration:            {min_duration:.2f} seconds")
    print(f"Max Duration:            {max_duration:.2f} seconds")
    print(f"Std Deviation:           {std_duration:.2f} seconds")
    print("-"*50)
    
    # Duration Buckets
    b_under_2 = sum(durations < 2.0)
    b_2_to_5 = sum((durations >= 2.0) & (durations < 5.0))
    b_5_to_10 = sum((durations >= 5.0) & (durations < 10.0))
    b_over_10 = sum(durations >= 10.0)
    
    print("Duration Distribution:")
    print(f"  < 2.0s:                {b_under_2:,} files ({b_under_2/len(durations)*100:.1f}%)")
    print(f"  2.0s - 5.0s:           {b_2_to_5:,} files ({b_2_to_5/len(durations)*100:.1f}%)")
    print(f"  5.0s - 10.0s:          {b_5_to_10:,} files ({b_5_to_10/len(durations)*100:.1f}%)")
    print(f"  >= 10.0s:              {b_over_10:,} files ({b_over_10/len(durations)*100:.1f}%)")
    print("-"*50)
    
    # Speaker stats
    num_speakers = len(speaker_durations)
    print(f"Total Speakers:          {num_speakers}")
    if num_speakers > 1 and "unknown" not in speaker_durations:
        sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 Speakers by Duration:")
        for spk, spk_dur in sorted_speakers[:10]:
            print(f"  - {spk:<20}: {format_time(spk_dur)} ({spk_dur:.2f}s)")
            
    print("="*50)

if __name__ == "__main__":
    main()
