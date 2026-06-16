#!/usr/bin/env python3
"""
Script to split the vie_neu dataset into 5 balanced partitions by speaker.
Balances partitions by total file size.
"""

import os
import shutil
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Split vie_neu dataset into 5 balanced folders.")
    parser.add_argument(
        "--src_dir", 
        type=str, 
        default="/home/reg/TTS_DATA/data/vie_neu",
        help="Path to the source vie_neu directory."
    )
    parser.add_argument(
        "--dest_prefix", 
        type=str, 
        default="/home/reg/TTS_DATA/data/vie_neu_part",
        help="Prefix path for destination partitions (e.g., vie_neu_part -> vie_neu_part_1...)"
    )
    parser.add_argument(
        "--num_parts", 
        type=int, 
        default=5,
        help="Number of partitions to split into."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["move", "copy", "symlink"],
        default="move",
        help="Method to distribute files: 'move' (default), 'copy', or 'symlink'."
    )
    return parser.parse_args()

def get_dir_size(path):
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
                file_count += 1
    return total_size, file_count

def main():
    args = parse_args()
    
    if not os.path.exists(args.src_dir):
        print(f"Error: Source directory '{args.src_dir}' does not exist.")
        return
        
    # Get all speaker folders (subdirectories)
    speaker_dirs = [
        d for d in os.listdir(args.src_dir) 
        if os.path.isdir(os.path.join(args.src_dir, d))
    ]
    
    if not speaker_dirs:
        print("No speaker directories found in source directory.")
        return
        
    print(f"Analyzing {len(speaker_dirs)} speaker directories for balance...")
    
    # Calculate sizes and file counts for each speaker
    speaker_data = []
    for spk in tqdm(speaker_dirs, desc="Analyzing size"):
        spk_path = os.path.join(args.src_dir, spk)
        size, count = get_dir_size(spk_path)
        speaker_data.append({
            "name": spk,
            "path": spk_path,
            "size": size,
            "count": count
        })
        
    # Sort speakers by size descending (for greedy partition)
    speaker_data.sort(key=lambda x: x["size"], reverse=True)
    
    # Initialize bins
    bins = [[] for _ in range(args.num_parts)]
    bin_sizes = [0 for _ in range(args.num_parts)]
    bin_counts = [0 for _ in range(args.num_parts)]
    
    # Greedy distribution (greedy heuristic for partition problem)
    for spk in speaker_data:
        # Find bin with smallest total size
        min_bin_idx = bin_sizes.index(min(bin_sizes))
        bins[min_bin_idx].append(spk)
        bin_sizes[min_bin_idx] += spk["size"]
        bin_counts[min_bin_idx] += spk["count"]
        
    print("\nPartition Plan:")
    for i in range(args.num_parts):
        size_gb = bin_sizes[i] / (1024 ** 3)
        print(f"  Part {i+1}: {len(bins[i])} speakers, {bin_counts[i]} files, {size_gb:.2f} GB")
        
    # Confirm actions
    print(f"\nDistributing files using '{args.mode}' mode...")
    
    # Create destination directories
    for i in range(args.num_parts):
        dest_dir = f"{args.dest_prefix}_{i+1}"
        os.makedirs(dest_dir, exist_ok=True)
        
        for spk in tqdm(bins[i], desc=f"Writing Part {i+1}"):
            src_spk_path = spk["path"]
            dest_spk_path = os.path.join(dest_dir, spk["name"])
            
            if args.mode == "move":
                shutil.move(src_spk_path, dest_spk_path)
            elif args.mode == "copy":
                shutil.copytree(src_spk_path, dest_spk_path, dirs_exist_ok=True)
            elif args.mode == "symlink":
                # Create a relative symlink to avoid broken paths if moved
                rel_src = os.path.relpath(src_spk_path, os.path.dirname(dest_spk_path))
                os.symlink(rel_src, dest_spk_path)
                
    print("\nDataset successfully split!")

if __name__ == "__main__":
    main()
