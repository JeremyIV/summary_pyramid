"""Functions for generating summaries in the summary pyramid approach."""

import os
import time
import shutil
import argparse
from typing import List, Tuple, Dict
import anthropic
import prompts
from chunk_document import chunk_document

def get_final_answer(
    final_summary: str,
    user_query: str,
    total_chunks: int,
    total_summary_levels: int,
    model: str = "claude-3-7-sonnet-20250219",
    context_window_size: int = 100000,
    tokens_per_selection: int = 8000,
    answer_token_limit: int = 4000
) -> str:
    """Generate a final answer to the user's query based on the top-level summary.
    
    This function takes the final summary from the summary pyramid and generates
    a comprehensive answer to the user's original query.
    
    Args:
        final_summary: The final top-level summary from the pyramid.
        user_query: The user's original query about the document.
        total_chunks: Total number of chunks in the document.
        total_summary_levels: The total number of summary levels in the pyramid.
        model: The model to use for generating the answer.
        context_window_size: The context window size of the model in tokens.
        tokens_per_selection: The target size for content selections in tokens.
        answer_token_limit: The maximum size for the final answer in tokens.
        
    Returns:
        The generated answer as a string.
    """
    # Create the client
    client = anthropic.Anthropic()
    
    # Generate the system prompt
    system_prompt = prompts.get_system_prompt(
        context_window_size=context_window_size,
        tokens_per_selection=tokens_per_selection,
        summary_token_limit=answer_token_limit
    )
    
    # Generate the user prompt
    user_prompt = prompts.get_final_answer_prompt(
        user_query=user_query,
        total_chunks=total_chunks,
        total_summary_levels=total_summary_levels,
        final_summary=final_summary
    )
    
    # Call the API
    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=answer_token_limit
    )
    
    # Extract the answer text from the response
    answer = response.content[0].text
    return answer


def get_base_summary(
    chunk_content: str,
    user_query: str,
    total_chunks: int,
    chunk_range_start: int,
    chunk_range_end: int,
    model: str = "claude-3-7-sonnet-20250219",
    context_window_size: int = 100000,
    tokens_per_selection: int = 8000,
    summary_token_limit: int = 2000
) -> str:
    """Generate a base-level summary for a content chunk.
    
    Args:
        chunk_content: The content to summarize.
        user_query: The user's query about the document.
        total_chunks: Total number of chunks in the document.
        chunk_range_start: Starting chunk number of this content.
        chunk_range_end: Ending chunk number of this content.
        model: The model to use for summarization.
        context_window_size: The context window size of the model in tokens.
        tokens_per_selection: The target size for content selections in tokens.
        summary_token_limit: The maximum size for summaries in tokens.
        
    Returns:
        The generated summary as a string.
    """
    # Create the client
    client = anthropic.Anthropic()
    
    # Generate the system prompt
    system_prompt = prompts.get_system_prompt(
        context_window_size=context_window_size,
        tokens_per_selection=tokens_per_selection,
        summary_token_limit=summary_token_limit
    )
    
    # Generate the user prompt
    user_prompt = prompts.get_base_summary_prompt(
        user_query=user_query,
        total_chunks=total_chunks,
        chunk_range_start=chunk_range_start,
        chunk_range_end=chunk_range_end,
        chunk_content=chunk_content
    )
    
    # Call the API
    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=summary_token_limit
    )
    
    # Extract the summary text from the response
    summary = response.content[0].text
    return summary


def get_base_summaries(
    chunks: List[str],
    user_query: str,
    window_size: int = 10,
    stride: int = 9,
    model: str = "claude-3-7-sonnet-20250219",
    context_window_size: int = 100000,
    tokens_per_selection: int = 8000,
    summary_token_limit: int = 1000
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Generate base-level summaries for all chunks in the document using a sliding window.
    
    Args:
        chunks: List of document chunks.
        user_query: The user's query about the document.
        window_size: Number of chunks to include in each window.
        stride: Number of chunks to advance the window on each step.
        model: The model to use for summarization.
        context_window_size: The context window size of the model in tokens.
        tokens_per_selection: The target size for content selections in tokens.
        summary_token_limit: The maximum size for summaries in tokens.
        
    Returns:
        A tuple containing:
        - List of generated summaries
        - List of tuples with (start_chunk, end_chunk) indices for each summary
    """
    total_chunks = len(chunks)
    summaries = []
    chunk_ranges = []
    
    # Process the document with a sliding window
    start_idx = 0
    while start_idx < total_chunks:
        # Calculate the end index for this window
        end_idx = min(start_idx + window_size - 1, total_chunks - 1)
        
        # Convert to 1-indexed chunk numbers for the prompt
        chunk_range_start = start_idx + 1
        chunk_range_end = end_idx + 1
        
        # Join the chunks in this window (simply concatenate them)
        window_content = "".join(chunks[start_idx:end_idx + 1])
        
        # Generate the summary for this window
        print(f"getting summary for chunks {chunk_range_start} thru {chunk_range_end}")
        summary = get_base_summary(
            chunk_content=window_content,
            user_query=user_query,
            total_chunks=total_chunks,
            chunk_range_start=chunk_range_start,
            chunk_range_end=chunk_range_end,
            model=model,
            context_window_size=context_window_size,
            tokens_per_selection=tokens_per_selection,
            summary_token_limit=summary_token_limit
        )
        
        # Store the summary and its metadata
        summaries.append(summary)
        chunk_ranges.append((chunk_range_start, chunk_range_end))
        
        # Move the window forward by the stride
        start_idx += stride
            
    return summaries, chunk_ranges


def get_recursive_summary(
    summaries: List[str],
    chunk_ranges: List[Tuple[int, int]],
    user_query: str,
    total_chunks: int,
    summary_level: int,
    summary_range_start: int,
    summary_range_end: int,
    total_summaries: int,
    model: str = "claude-3-7-sonnet-20250219",
    context_window_size: int = 100000,
    tokens_per_selection: int = 8000,
    summary_token_limit: int = 2000
) -> str:
    """Generate a recursive summary from a set of lower-level summaries.
    
    Args:
        summaries: List of summaries to combine.
        chunk_ranges: List of tuples with (start_chunk, end_chunk) indices for each summary.
        user_query: The user's query about the document.
        total_chunks: Total number of chunks in the document.
        summary_level: The current level of summarization (1, 2, etc.).
        summary_range_start: The starting summary number in this batch.
        summary_range_end: The ending summary number in this batch.
        total_summaries: The total number of summaries at this level.
        model: The model to use for summarization.
        context_window_size: The context window size of the model in tokens.
        tokens_per_selection: The target size for content selections in tokens.
        summary_token_limit: The maximum size for summaries in tokens.
        
    Returns:
        The combined summary as a string.
    """
    # Create the client
    client = anthropic.Anthropic()
    
    # Generate the system prompt
    system_prompt = prompts.get_system_prompt(
        context_window_size=context_window_size,
        tokens_per_selection=tokens_per_selection,
        summary_token_limit=summary_token_limit
    )
    
    # Determine the chunk range covered by these summaries
    chunk_range_start = chunk_ranges[0][0]
    chunk_range_end = chunk_ranges[-1][1]
    
    # Create a list of summary objects for the template
    summary_objects = []
    for i, (summary, (start, end)) in enumerate(zip(summaries, chunk_ranges)):
        summary_objects.append({
            "start_chunk": start,
            "end_chunk": end,
            "content": summary
        })
    
    # Generate the user prompt
    user_prompt = prompts.get_recursive_summary_prompt(
        user_query=user_query,
        total_chunks=total_chunks,
        summary_level=summary_level,
        summary_range_start=summary_range_start,
        summary_range_end=summary_range_end,
        total_summaries=total_summaries,
        chunk_range_start=chunk_range_start,
        chunk_range_end=chunk_range_end,
        summaries=summary_objects
    )
    
    # Call the API
    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=summary_token_limit
    )
    
    # Extract the summary text from the response
    summary = response.content[0].text
    return summary


def get_recursive_summaries(
    summaries: List[str],
    chunk_ranges: List[Tuple[int, int]],
    user_query: str,
    total_chunks: int,
    summary_level: int,
    window_size: int = 5,
    stride: int = 4,
    model: str = "claude-3-7-sonnet-20250219",
    context_window_size: int = 100000,
    tokens_per_selection: int = 8000,
    summary_token_limit: int = 2000
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Generate higher-level summaries from a set of lower-level summaries using a sliding window.
    
    Args:
        summaries: List of summaries to combine.
        chunk_ranges: List of tuples with (start_chunk, end_chunk) indices for each summary.
        user_query: The user's query about the document.
        total_chunks: Total number of chunks in the document.
        summary_level: The level of the input summaries (output will be level+1).
        window_size: Number of summaries to include in each window.
        stride: Number of summaries to advance the window on each step.
        model: The model to use for summarization.
        context_window_size: The context window size of the model in tokens.
        tokens_per_selection: The target size for content selections in tokens.
        summary_token_limit: The maximum size for summaries in tokens.
        
    Returns:
        A tuple containing:
        - List of higher-level summaries
        - List of tuples with (start_chunk, end_chunk) indices for each summary
    """
    total_summaries = len(summaries)
    new_summaries = []
    new_chunk_ranges = []
    
    # Process the summaries with a sliding window
    start_idx = 0
    while start_idx < total_summaries:
        # Calculate the end index for this window
        end_idx = min(start_idx + window_size - 1, total_summaries - 1)
        
        # Get the window of summaries
        window_summaries = summaries[start_idx:end_idx + 1]
        window_chunk_ranges = chunk_ranges[start_idx:end_idx + 1]
        
        # Summaries are 1-indexed for the prompt
        summary_range_start = start_idx + 1
        summary_range_end = end_idx + 1
        
        print(f"Generating level {summary_level+1} summary for summaries {summary_range_start}-{summary_range_end}")
        
        # Generate the recursive summary for this window
        recursive_summary = get_recursive_summary(
            summaries=window_summaries,
            chunk_ranges=window_chunk_ranges,
            user_query=user_query,
            total_chunks=total_chunks,
            summary_level=summary_level,
            summary_range_start=summary_range_start,
            summary_range_end=summary_range_end,
            total_summaries=total_summaries,
            model=model,
            context_window_size=context_window_size,
            tokens_per_selection=tokens_per_selection,
            summary_token_limit=summary_token_limit
        )
        
        # The new chunk range covers all chunks in the window
        new_chunk_start = window_chunk_ranges[0][0]
        new_chunk_end = window_chunk_ranges[-1][1]
        
        # Store the summary and its metadata
        new_summaries.append(recursive_summary)
        new_chunk_ranges.append((new_chunk_start, new_chunk_end))
        
        # Move the window forward by the stride
        start_idx += stride
    
    return new_summaries, new_chunk_ranges


def get_summary_pyramid(
    chunks: List[str],
    user_query: str,
    base_window_size: int = 10,
    base_stride: int = 9,
    recursive_window_size: int = 5,
    recursive_stride: int = 4,
    model: str = "claude-3-7-sonnet-20250219",
    context_window_size: int = 100000,
    tokens_per_selection: int = 8000,
    summary_token_limit: int = 2000
) -> List[Tuple[List[str], List[Tuple[int, int]]]]:
    """Generate a complete summary pyramid from document chunks to a single top-level summary.
    
    This function builds a hierarchical pyramid of summaries, starting with base-level
    summaries of the document chunks, and recursively combining them into higher-level
    summaries until there is a single summary covering the entire document.
    
    Args:
        chunks: List of document chunks.
        user_query: The user's query about the document.
        base_window_size: Window size for the base summaries.
        base_stride: Stride for the base summaries.
        recursive_window_size: Window size for the recursive summaries.
        recursive_stride: Stride for the recursive summaries.
        model: The model to use for summarization.
        context_window_size: The context window size of the model in tokens.
        tokens_per_selection: The target size for content selections in tokens.
        summary_token_limit: The maximum size for summaries in tokens.
        
    Returns:
        A list of tuples, where each tuple contains:
        - List of summaries at that level
        - List of tuples with (start_chunk, end_chunk) indices for each summary
        
        The first element is the base level summaries, and each subsequent element
        is a higher level of recursive summarization. The final element will contain
        a single summary covering the entire document.
    """
    total_chunks = len(chunks)
    pyramid = []
    
    # Generate base-level summaries
    print(f"Generating level 1 summaries (base level)...")
    base_summaries, base_chunk_ranges = get_base_summaries(
        chunks=chunks,
        user_query=user_query,
        window_size=base_window_size,
        stride=base_stride,
        model=model,
        context_window_size=context_window_size,
        tokens_per_selection=tokens_per_selection,
        summary_token_limit=summary_token_limit
    )
    
    # Add base level to the pyramid
    pyramid.append((base_summaries, base_chunk_ranges))
    
    # Generate recursive summaries until we reach a single summary
    current_summaries = base_summaries
    current_chunk_ranges = base_chunk_ranges
    current_level = 1
    
    # Keep generating higher-level summaries until we have just one summary
    while len(current_summaries) > 1:
        current_level += 1
        print(f"\nGenerating level {current_level} summaries...")
        
        # Generate the next level of summaries
        next_summaries, next_chunk_ranges = get_recursive_summaries(
            summaries=current_summaries,
            chunk_ranges=current_chunk_ranges,
            user_query=user_query,
            total_chunks=total_chunks,
            summary_level=current_level - 1,
            window_size=recursive_window_size,
            stride=recursive_stride,
            model=model,
            context_window_size=context_window_size,
            tokens_per_selection=tokens_per_selection,
            summary_token_limit=summary_token_limit
        )
        
        # Add this level to the pyramid
        pyramid.append((next_summaries, next_chunk_ranges))
        
        # Update for next iteration
        current_summaries = next_summaries
        current_chunk_ranges = next_chunk_ranges
        
        print(f"Level {current_level} complete: {len(current_summaries)} summaries")
    
    return pyramid


def main():
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
    
    # Generate the summary pyramid
    pyramid_start_time = time.time()
    
    pyramid = get_summary_pyramid(
        chunks=chunks,
        user_query=query,
        base_window_size=args.window_size,
        base_stride=args.stride,
        recursive_window_size=args.window_size,
        recursive_stride=args.stride,
        model=args.model,
        context_window_size=args.context_window,
        tokens_per_selection=args.tokens_per_selection,
        summary_token_limit=args.summary_token_limit
    )
    
    pyramid_elapsed_time = time.time() - pyramid_start_time
    print(f"\nPyramid generation completed in {pyramid_elapsed_time:.2f} seconds")
    print(f"Generated {len(pyramid)} levels of summaries")
    
    # Write out all summaries to the directory structure
    for level, (summaries, chunk_ranges) in enumerate(pyramid, 1):
        # Create a directory for this level
        level_dir = os.path.join(args.output_dir, f"level_{level}")
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
    metadata_path = os.path.join(args.output_dir, "pyramid_metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"Document: {args.document}\n")
        f.write(f"Query: {query}\n")
        f.write(f"Total document chunks: {total_chunks}\n")
        f.write(f"Tokens per chunk: {args.tokens_per_chunk}\n")
        f.write(f"Tokens per selection: {args.tokens_per_selection}\n")
        f.write(f"Window size: {args.window_size}, stride: {args.stride}\n")
        f.write(f"Total levels in pyramid: {len(pyramid)}\n\n")
        
        f.write("Pyramid structure:\n")
        for level, (summaries, chunk_ranges) in enumerate(pyramid, 1):
            f.write(f"Level {level}: {len(summaries)} summaries\n")
            for i, (start_chunk, end_chunk) in enumerate(chunk_ranges):
                f.write(f"  Summary {i+1}: Chunks {start_chunk}-{end_chunk}\n")
    
    print(f"\nSummary pyramid has been written to {args.output_dir}/")
    print(f"Metadata written to {metadata_path}")
    
    # If the pyramid has a single top-level summary, generate the final answer
    if len(pyramid[-1][0]) == 1:
        print("\nGenerating final answer...")
        final_summary = pyramid[-1][0][0]
        
        answer_start_time = time.time()
        answer = get_final_answer(
            final_summary=final_summary,
            user_query=query,
            total_chunks=total_chunks,
            total_summary_levels=len(pyramid),
            model=args.model,
            context_window_size=args.context_window,
            tokens_per_selection=args.tokens_per_selection,
            answer_token_limit=args.answer_token_limit
        )
        answer_elapsed_time = time.time() - answer_start_time
        
        # Write the final answer to a file
        answer_path = os.path.join(args.output_dir, "final_answer.txt")
        with open(answer_path, "w", encoding="utf-8") as f:
            f.write(answer)
        
        print(f"Final answer generated in {answer_elapsed_time:.2f} seconds")
        print(f"Answer written to {answer_path}")
        
        # Also write the final summary for reference
        summary_path = os.path.join(args.output_dir, "final_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(final_summary)
        
        print(f"Final top-level summary written to {summary_path}")
    else:
        print("\nWarning: Pyramid did not converge to a single summary, no final answer generated.")
    
    total_elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()