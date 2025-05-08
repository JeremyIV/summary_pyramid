"""Test script for the base summaries functionality."""

from chunk_document import chunk_document
from summary_pyramid import get_base_summaries

# Read the document
with open("documents/mobydick.txt", "r", encoding="utf-8") as f:
    document = f.read()

# Read the query
with open("queries/query.txt", "r", encoding="utf-8") as f:
    query = f.read().strip()

print(f"Document length: {len(document)} characters")
print(f"Query: {query}")

# Chunk the document
tokens_per_chunk = 1000
chunks = chunk_document(document, tokens_per_chunk)
total_chunks = len(chunks)

print(f"Document split into {total_chunks} chunks")

# Generate summaries using sliding window
window_size = 3 
stride = 2
print(f"Using window size {window_size} and stride {stride}")

# Generate base summaries
summaries, chunk_ranges = get_base_summaries(
    chunks=chunks,
    user_query=query,
    window_size=window_size,
    stride=stride,
    model="claude-3-5-haiku-latest",
)

print(f"Generated {len(summaries)} summaries")

# Write the summaries to files
for i, (summary, chunk_range) in enumerate(zip(summaries, chunk_ranges)):
    start, end = chunk_range
    filename = f"test_summary_{i+1}_chunks_{start}-{end}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"Wrote summary {i+1} (chunks {start}-{end}) to {filename}")

# Also write a summary of the ranges
with open("test_summaries_ranges.txt", "w", encoding="utf-8") as f:
    f.write(f"Total document chunks: {total_chunks}\n")
    f.write(f"Window size: {window_size}, Stride: {stride}\n\n")
    f.write("Summary coverage:\n")
    
    for i, (start, end) in enumerate(chunk_ranges):
        f.write(f"Summary {i+1}: Chunks {start}-{end}\n")

print("Summary ranges written to test_summaries_ranges.txt")