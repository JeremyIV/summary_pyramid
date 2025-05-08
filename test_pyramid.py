"""Test script for the summary pyramid.

This script tests the complete summary pyramid functionality.
It reads a document, creates a hierarchical pyramid of summaries,
and organizes the outputs in a structured directory format.
"""

import os
import time
import shutil
from chunk_document import chunk_document
from summary_pyramid import get_summary_pyramid

# Create output directory structure
output_dir = "pyramid_output"
if os.path.exists(output_dir):
    # Clear any existing output
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Read the document
with open("documents/mobydick.txt", "r", encoding="utf-8") as f:
    document = f.read()

# Read the query
with open("queries/query.txt", "r", encoding="utf-8") as f:
    query = f.read().strip()

print(f"Document length: {len(document)} characters")
print(f"Query: {query}")

# Chunk the document - using smaller chunks for faster testing
tokens_per_chunk = 5000
chunks = chunk_document(document, tokens_per_chunk)
total_chunks = len(chunks)

print(f"Document split into {total_chunks} chunks of approximately {tokens_per_chunk} tokens each")

# Configure pyramid parameters - smaller values for faster testing
base_window_size = 3
base_stride = 2
recursive_window_size = 3
recursive_stride = 2

print(f"Using base window size {base_window_size}, stride {base_stride}")
print(f"Using recursive window size {recursive_window_size}, stride {recursive_stride}")

# Generate the summary pyramid
start_time = time.time()

pyramid = get_summary_pyramid(
    chunks=chunks,
    user_query=query,
    base_window_size=base_window_size,
    base_stride=base_stride,
    recursive_window_size=recursive_window_size,
    recursive_stride=recursive_stride
)

elapsed_time = time.time() - start_time
print(f"\nPyramid generation completed in {elapsed_time:.2f} seconds")
print(f"Generated {len(pyramid)} levels of summaries")

# Write out all summaries to the directory structure
for level, (summaries, chunk_ranges) in enumerate(pyramid, 1):
    # Create a directory for this level
    level_dir = os.path.join(output_dir, f"level_{level}")
    os.makedirs(level_dir, exist_ok=True)
    
    print(f"\nWriting {len(summaries)} summaries for level {level}")
    
    # Write each summary to a file named by its chunk range
    for i, (summary, (start_chunk, end_chunk)) in enumerate(zip(summaries, chunk_ranges)):
        # Create the filename based on the chunk range
        filename = f"chunks_{start_chunk}-{end_chunk}.txt"
        filepath = os.path.join(level_dir, filename)
        
        # Write the summary to the file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"  Wrote summary {i+1}: chunks {start_chunk}-{end_chunk}")

# Create a metadata file with information about the pyramid
metadata_path = os.path.join(output_dir, "pyramid_metadata.txt")
with open(metadata_path, "w", encoding="utf-8") as f:
    f.write(f"Document: documents/mobydick.txt\n")
    f.write(f"Query: {query}\n")
    f.write(f"Total document chunks: {total_chunks}\n")
    f.write(f"Tokens per chunk: {tokens_per_chunk}\n")
    f.write(f"Base window size: {base_window_size}, stride: {base_stride}\n")
    f.write(f"Recursive window size: {recursive_window_size}, stride: {recursive_stride}\n")
    f.write(f"Total levels in pyramid: {len(pyramid)}\n\n")
    
    f.write("Pyramid structure:\n")
    for level, (summaries, chunk_ranges) in enumerate(pyramid, 1):
        f.write(f"Level {level}: {len(summaries)} summaries\n")
        for i, (start_chunk, end_chunk) in enumerate(chunk_ranges):
            f.write(f"  Summary {i+1}: Chunks {start_chunk}-{end_chunk}\n")

print(f"\nSummary pyramid has been written to {output_dir}/")
print(f"Metadata written to {metadata_path}")

# If the pyramid has a single top-level summary, write it to a separate file for easy access
if len(pyramid[-1][0]) == 1:
    final_summary = pyramid[-1][0][0]
    final_summary_path = os.path.join(output_dir, "final_summary.txt")
    
    with open(final_summary_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    
    print(f"Final top-level summary written to {final_summary_path}")