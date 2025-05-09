#!/usr/bin/env python3
"""
Summary Rollup: A tool for answering questions about very long documents 
using a recurrent summarization approach.
"""

import os
import time
import shutil
import argparse
from typing import List, Tuple, Dict
import anthropic
import recurrent_prompts
from chunk_document import chunk_document

###########################################################################
## SETUP
###########################################################################

parser = argparse.ArgumentParser(description="Generate a sequential summary rollup for a document and answer queries")

# Input document and query
parser.add_argument("--document", default="documents/mobydick.txt", help="Path to the document file")
parser.add_argument("--query", default="queries/query.txt", help="Path to query file or direct query string")

# Model and token parameters
parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Claude model to use")
parser.add_argument("--context-window", type=int, default=100000, help="Context window size in tokens")
parser.add_argument("--tokens-per-selection", type=int, default=5000, help="Target size for content selections in tokens")
parser.add_argument("--tokens-per-chunk", type=int, default=1000, help="Size to chunk the document into")
parser.add_argument("--summary-token-limit", type=int, default=2000, help="Maximum token limit for summaries")
parser.add_argument("--answer-token-limit", type=int, default=4000, help="Maximum token limit for final answer")

# Output parameters
parser.add_argument("--output-dir", default="rollup_output", help="Directory to store outputs")
parser.add_argument("--clear-output", action="store_true", help="Clear output directory if it exists")

args = parser.parse_args()

# Set up output directory
if args.clear_output and os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

# Read the document
with open(args.document, "r", encoding="utf-8") as f:
    document = f.read()

# Read the query - either from file or directly from command line
if os.path.exists(args.query):
    with open(args.query, "r", encoding="utf-8") as f:
        query = f.read().strip()
else:
    query = args.query.strip()

print(f"Document length: {len(document)} characters")
print(f"Query: {query}")

# Chunk the document
start_time = time.time()
chunks = chunk_document(document, args.tokens_per_chunk, args.model)
total_chunks = len(chunks)

print(f"Document chunked into {total_chunks} chunks of approximately {args.tokens_per_chunk} tokens each")
print(f"Chunking took {time.time() - start_time:.2f} seconds")

# Initialize system prompt and client
system_prompt = recurrent_prompts.get_system_prompt(
    context_window_size=args.context_window,
    tokens_per_selection=args.tokens_per_selection,
    summary_token_limit=args.summary_token_limit
)

client = anthropic.Anthropic()

###########################################################################
## SUMMARY ROLLUP
###########################################################################

rollup_start_time = time.time()
print(f"\nStarting sequential summary rollup process")

# Process chunks sequentially
summary_history = []  # Track all intermediate summaries

# Step 1: Process the first chunk to get initial summary
print(f"Processing chunk 1 of {total_chunks}")
initial_prompt = recurrent_prompts.get_base_summary_prompt(
    user_query=query,
    total_chunks=total_chunks,
    chunk_content=chunks[0],
    summary_token_limit=args.summary_token_limit
)

response = client.messages.create(
    model=args.model,
    system=system_prompt,
    messages=[{"role": "user", "content": initial_prompt}],
    max_tokens=args.summary_token_limit
)

current_summary = response.content[0].text
summary_history.append(current_summary)

print(f"Generated initial summary")

# Step 2: Process each subsequent chunk, updating the summary
current_chunk = 2  # Start with the second chunk
while current_chunk <= total_chunks:
    print(f"Processing chunk {current_chunk} of {total_chunks}")
    
    recursive_prompt = recurrent_prompts.get_recursive_summary_prompt(
        user_query=query,
        total_chunks=total_chunks,
        current_chunk=current_chunk,
        chunks_processed=current_chunk,
        current_summary=current_summary,
        new_chunk_content=chunks[current_chunk-1],  # 0-indexed array
        summary_token_limit=args.summary_token_limit
    )
    
    response = client.messages.create(
        model=args.model,
        system=system_prompt,
        messages=[{"role": "user", "content": recursive_prompt}],
        max_tokens=args.summary_token_limit
    )
    
    current_summary = response.content[0].text
    summary_history.append(current_summary)
    
    current_chunk += 1

rollup_elapsed_time = time.time() - rollup_start_time
print(f"\nSummary rollup completed in {rollup_elapsed_time:.2f} seconds")
print(f"Processed {total_chunks} chunks sequentially")

###########################################################################
## GET THE FINAL ANSWER
###########################################################################

answer_start_time = time.time()
print("\nGenerating final answer based on complete summary")

# Use the same system prompt, but create a final answer prompt
answer_prompt = f"""<USER_QUERY>
{query}
</USER_QUERY>

<DOCUMENT_INFO total_chunks="{total_chunks}" />

Below is a comprehensive summary of the entire document:

<FINAL_SUMMARY>
{current_summary}
</FINAL_SUMMARY>

Based on the summary above, please provide a detailed answer to the user's query. 
Focus on being accurate, comprehensive, and directly addressing the question asked.
"""

response = client.messages.create(
    model=args.model,
    system=system_prompt,
    messages=[{"role": "user", "content": answer_prompt}],
    max_tokens=args.answer_token_limit
)

answer = response.content[0].text
answer_elapsed_time = time.time() - answer_start_time

###########################################################################
## SAVE OUTPUT
###########################################################################

# Create directory for summary stages
summaries_dir = os.path.join(args.output_dir, "summaries")
os.makedirs(summaries_dir, exist_ok=True)

# Write all intermediate summaries
for i, summary in enumerate(summary_history, 1):
    filename = f"summary_stage_{i}_of_{len(summary_history)}.txt"
    filepath = os.path.join(summaries_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"Wrote summary stage {i} to {filepath}")

# Write the final answer
answer_path = os.path.join(args.output_dir, "final_answer.txt")
with open(answer_path, "w", encoding="utf-8") as f:
    f.write(answer)

# Write the final summary for reference
summary_path = os.path.join(args.output_dir, "final_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(current_summary)

# Create metadata file
metadata_path = os.path.join(args.output_dir, "rollup_metadata.txt")
with open(metadata_path, "w", encoding="utf-8") as f:
    f.write(f"Document: {args.document}\n")
    f.write(f"Query: {query}\n")
    f.write(f"Total document chunks: {total_chunks}\n")
    f.write(f"Tokens per chunk: {args.tokens_per_chunk}\n")
    f.write(f"Tokens per selection: {args.tokens_per_selection}\n")
    f.write(f"Summary token limit: {args.summary_token_limit}\n")
    f.write(f"Total summary stages: {len(summary_history)}\n\n")
    
    f.write("Processing timeline:\n")
    for i in range(total_chunks):
        if i == 0:
            f.write(f"Stage 1: Initial summary of chunk 1\n")
        else:
            f.write(f"Stage {i+1}: Updated summary incorporating chunk {i+1}\n")

print(f"\nFinal answer generated in {answer_elapsed_time:.2f} seconds")
print(f"Answer written to {answer_path}")
print(f"Final summary written to {summary_path}")
print(f"Metadata written to {metadata_path}")

total_elapsed_time = time.time() - start_time
print(f"\nTotal processing time: {total_elapsed_time:.2f} seconds")