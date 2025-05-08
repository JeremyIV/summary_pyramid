"""Test script for the recursive summary functionality."""

import os
import glob
import re
from typing import List, Tuple

from summary_pyramid import get_recursive_summary

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
print(f"Found {len(summary_files)} summary files")

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

# Generate the recursive summary
if summaries:
    print("\nGenerating recursive summary...")
    recursive_summary = get_recursive_summary(
        summaries=summaries,
        chunk_ranges=chunk_ranges,
        user_query=query,
        total_chunks=total_chunks,
        summary_level=1,  # Level of input summaries
        summary_range_start=1,
        summary_range_end=len(summaries),
        total_summaries=len(summaries)
    )
    
    # Write the recursive summary to a file
    output_file = "test_recursive_summary.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(recursive_summary)
    
    print(f"Recursive summary written to {output_file}")
    
    # Also write metadata about the summary
    with open("test_recursive_summary_metadata.txt", "w", encoding="utf-8") as f:
        f.write(f"Total document chunks: {total_chunks}\n")
        f.write(f"Input summary level: 1\n")
        f.write(f"Number of input summaries: {len(summaries)}\n")
        f.write(f"Chunk range: {chunk_ranges[0][0]}-{chunk_ranges[-1][1]}\n")
    
    print(f"Metadata written to test_recursive_summary_metadata.txt")
else:
    print("No summaries found to process.")