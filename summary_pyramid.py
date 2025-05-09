import os
import time
import shutil
import argparse
from typing import List, Tuple, Dict
import anthropic
import prompts
from chunk_document import chunk_document

###########################################################################
## SETUP
###########################################################################

"""Main function for command line usage of the summary pyramid system."""
parser = argparse.ArgumentParser(description="Generate a summary pyramid for a document and answer queries")

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

# Sliding window parameters
parser.add_argument("--window-size", type=int, default=5, help="Window size for all summary levels")
parser.add_argument("--stride", type=int, default=4, help="Stride for all summary levels")

# Output parameters
parser.add_argument("--output-dir", default="pyramid_output", help="Directory to store outputs")
parser.add_argument("--clear-output", action="store_true", help="Clear output directory if it exists")

args = parser.parse_args()

# Ensure stride is reasonable (to ensure we make progress)
assert args.stride > 0, "Stride must be at least 1"

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

# Configure pyramid parameters
print(f"Using window size {args.window_size}, stride {args.stride} for all summary levels")

# Initialize system prompt
system_prompt = prompts.get_system_prompt(
    context_window_size=args.context_window,
    tokens_per_selection=args.tokens_per_selection,
    summary_token_limit=args.summary_token_limit
)

client = anthropic.Anthropic()

###########################################################################
## SUMMARY PYRAMID
###########################################################################

pyramid_start_time = time.time()

# Initialize with first level: (summary content, start_chunk_index, end_chunk_index)
current_summaries = [(chunk, idx, idx) for idx, chunk in enumerate(chunks)]
summary_pyramid = [current_summaries]
current_level = 1

print(f"Building pyramid level {current_level} with {len(current_summaries)} summaries")

while len(current_summaries) > 1:
    current_level += 1
    print(f"Building pyramid level {current_level}")
    next_summaries = []
    window_start_idx = 0
    
    while window_start_idx < len(current_summaries):
        window_end_idx = min(window_start_idx + args.window_size, len(current_summaries))
        
        summaries_in_window = current_summaries[window_start_idx:window_end_idx]
        if not summaries_in_window:  # Safeguard against empty windows
            break
            
        new_summary_start_chunk = summaries_in_window[0][1]  # Access start_chunk_index
        new_summary_end_chunk = summaries_in_window[-1][2]   # Access end_chunk_index
        
        print(f"  Processing window for chunks {new_summary_start_chunk+1}-{new_summary_end_chunk+1}")
        
        if len(summary_pyramid) == 1:
            # For base level, collect the actual text chunks
            chunk_text = "".join([s[0] for s in summaries_in_window])
            user_prompt = prompts.get_base_summary_prompt(
                user_query=query,
                total_chunks=total_chunks,
                chunk_range_start=new_summary_start_chunk+1,  # Convert to 1-based indexing
                chunk_range_end=new_summary_end_chunk+1,      # Convert to 1-based indexing
                chunk_content=chunk_text
            )
        else:
            # For higher levels, use the summaries from previous level
            summary_objects = []
            for summary_text, start, end in summaries_in_window:
                summary_objects.append({
                    "start_chunk": start+1,  # Convert to 1-based indexing
                    "end_chunk": end+1,      # Convert to 1-based indexing
                    "content": summary_text
                })

            user_prompt = prompts.get_recursive_summary_prompt(
                user_query=query,
                total_chunks=total_chunks,
                summary_level=current_level-1,  # Current level of input summaries
                summary_range_start=window_start_idx+1,
                summary_range_end=window_end_idx,
                total_summaries=len(current_summaries),
                chunk_range_start=new_summary_start_chunk+1,  # Convert to 1-based indexing
                chunk_range_end=new_summary_end_chunk+1,      # Convert to 1-based indexing
                summaries=summary_objects
            )

        response = client.messages.create(
            model=args.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=args.summary_token_limit
        )
        
        new_summary_text = response.content[0].text
        next_summaries.append((new_summary_text, new_summary_start_chunk, new_summary_end_chunk))
        
        window_start_idx += args.stride
    
    print(f"  Created {len(next_summaries)} summaries at level {current_level}")
    current_summaries = next_summaries
    summary_pyramid.append(current_summaries)

pyramid_elapsed_time = time.time() - pyramid_start_time
print(f"\nPyramid generation completed in {pyramid_elapsed_time:.2f} seconds")
print(f"Generated {len(summary_pyramid)} levels of summaries")

###########################################################################
## GET THE FINAL ANSWER
###########################################################################

if not current_summaries:
    print("Error: No summaries generated. Cannot create final answer.")
    exit(1)

final_summary = current_summaries[0][0]
answer_start_time = time.time()

user_prompt = prompts.get_final_answer_prompt(
    user_query=query,
    total_chunks=total_chunks,
    total_summary_levels=len(summary_pyramid),
    final_summary=final_summary
)

# Call the API
response = client.messages.create(
    model=args.model,
    system=system_prompt,
    messages=[{"role": "user", "content": user_prompt}],
    max_tokens=args.answer_token_limit
)

# Extract the answer text from the response
answer = response.content[0].text
answer_elapsed_time = time.time() - answer_start_time

###########################################################################
## SAVE OUTPUT
###########################################################################

# Write out all summaries to the directory structure
for level, summaries in enumerate(summary_pyramid, 1):  # Start numbering from 1
    # Create a directory for this level
    level_dir = os.path.join(args.output_dir, f"level_{level}")
    os.makedirs(level_dir, exist_ok=True)
    
    print(f"\nWriting {len(summaries)} summaries for level {level}")
    
    # Write each summary to a file named by its chunk range
    for i, (summary, start_chunk, end_chunk) in enumerate(summaries, 1):  # Start numbering from 1
        # Create the filename based on the chunk range - convert to 1-based indexing for filenames
        filename = f"chunks_{start_chunk+1}-{end_chunk+1}.txt"
        filepath = os.path.join(level_dir, filename)
        
        # Write the summary to the file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"  Wrote summary {i}: chunks {start_chunk+1}-{end_chunk+1}")

# Write the final answer to a file
answer_path = os.path.join(args.output_dir, "final_answer.txt")
with open(answer_path, "w", encoding="utf-8") as f:
    f.write(answer)

# Also write the final summary for reference
summary_path = os.path.join(args.output_dir, "final_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(final_summary)

print(f"\nFinal answer generated in {answer_elapsed_time:.2f} seconds")
print(f"Answer written to {answer_path}")
print(f"Final summary written to {summary_path}")

total_elapsed_time = time.time() - start_time
print(f"\nTotal processing time: {total_elapsed_time:.2f} seconds")
