"""Test script for the recursive summaries functionality."""

import os
import glob
import re
from typing import List, Tuple

from summary_pyramid import get_recursive_summaries

# Read the query
with open("queries/query.txt", "r", encoding="utf-8") as f:
    query = f.read().strip()

# Get the total number of chunks from the ranges file
with open("test_summaries_ranges.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    total_chunks_line = lines[0]
    total_chunks = int(total_chunks_line.split(":", 1)[1].strip())

print(f"Total document chunks: {total_chunks}")
print(f"Query: {query}")

# Find all the summary files
summary_files = glob.glob("test_summary_*_chunks_*.txt")
print(f"Found {len(summary_files)} base summary files")

# Parse the summaries and their chunk ranges
summaries = []
chunk_ranges = []

# Extract the summary number and chunk range from filenames
pattern = r"test_summary_(\d+)_chunks_(\d+)-(\d+)\.txt"

for file_path in sorted(summary_files, key=lambda x: int(re.search(pattern, x).group(1))):
    match = re.search(pattern, file_path)
    if match:
        summary_num = int(match.group(1))
        chunk_start = int(match.group(2))
        chunk_end = int(match.group(3))
        
        # Read the summary content
        with open(file_path, "r", encoding="utf-8") as f:
            summary_content = f.read()
        
        summaries.append(summary_content)
        chunk_ranges.append((chunk_start, chunk_end))
        print(f"Loaded summary {summary_num}: Chunks {chunk_start}-{chunk_end}")

# Generate the recursive summaries with sliding window
if summaries:
    print("\nGenerating recursive summaries...")
    
    # Parameters for sliding window
    window_size = 3
    stride = 2
    summary_level = 1  # Level of input summaries
    
    # Generate recursive summaries
    level2_summaries, level2_chunk_ranges = get_recursive_summaries(
        summaries=summaries,
        chunk_ranges=chunk_ranges,
        user_query=query,
        total_chunks=total_chunks,
        summary_level=summary_level,
        window_size=window_size,
        stride=stride
    )
    
    print(f"\nGenerated {len(level2_summaries)} level 2 summaries")
    
    # Write the recursive summaries to files
    for i, (summary, chunk_range) in enumerate(zip(level2_summaries, level2_chunk_ranges)):
        start, end = chunk_range
        filename = f"test_level2_summary_{i+1}_chunks_{start}-{end}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"Wrote level 2 summary {i+1} (chunks {start}-{end}) to {filename}")
    
    # Also write metadata about the summaries
    with open("test_level2_summaries_metadata.txt", "w", encoding="utf-8") as f:
        f.write(f"Total document chunks: {total_chunks}\n")
        f.write(f"Input summary level: {summary_level}\n")
        f.write(f"Number of level 2 summaries: {len(level2_summaries)}\n")
        f.write(f"Window size: {window_size}, Stride: {stride}\n\n")
        f.write("Level 2 summary coverage:\n")
        
        for i, (start, end) in enumerate(level2_chunk_ranges):
            f.write(f"Summary {i+1}: Chunks {start}-{end}\n")
    
    print(f"Metadata written to test_level2_summaries_metadata.txt")
else:
    print("No summaries found to process.")