"""Test script for the summary pyramid."""

from chunk_document import chunk_document
from summary_pyramid import get_base_summary

# Read the document
with open("documents/mobydick.txt", "r", encoding="utf-8") as f:
    document = f.read()

# Read the query
with open("queries/query.txt", "r", encoding="utf-8") as f:
    query = f.read().strip()

print(f"Document length: {len(document)} characters")
print(f"Query: {query}")

# Chunk the document
tokens_per_chunk = 10000
chunks = chunk_document(document, tokens_per_chunk)
total_chunks = len(chunks)

print(f"Document split into {total_chunks} chunks")

# Generate summary for the first chunk
summary = get_base_summary(
    chunk_content=chunks[0],
    user_query=query,
    total_chunks=total_chunks,
    chunk_range_start=1,
    chunk_range_end=1
)

# Write the summary to a file
with open("test_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("Summary written to test_summary.txt")