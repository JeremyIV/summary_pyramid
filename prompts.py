"""Functions for rendering prompt templates for the summary pyramid.

This module contains functions that render Jinja templates into
prompts that can be sent to the AI model.
"""

import os
from jinja2 import Environment, FileSystemLoader

# Set up Jinja environment
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_templates")
env = Environment(loader=FileSystemLoader(template_dir))


def get_system_prompt(context_window_size, tokens_per_selection=10_000, summary_token_limit=1_000):
    """Render the system prompt template with the given context window size.
    
    Args:
        context_window_size (int): The size of the AI model's context window in tokens.
        tokens_per_selection (int): The size of each content selection in tokens.
        summary_token_limit (int): The maximum size for each summary in tokens.
        
    Returns:
        str: The rendered system prompt.
    """
    template = env.get_template("explanation.jinja")
    return template.render(
        context_window_size=context_window_size,
        tokens_per_selection=tokens_per_selection,
        summary_token_limit=summary_token_limit
    )


def get_base_summary_prompt(user_query, total_chunks, chunk_range_start, chunk_range_end, chunk_content):
    """Render the base summary prompt template.
    
    Args:
        user_query (str): The user's original query about the document.
        total_chunks (int): The total number of chunks in the document.
        chunk_range_start (int): The starting chunk number of this content section.
        chunk_range_end (int): The ending chunk number of this content section.
        chunk_content (str): The actual content to be summarized.
        
    Returns:
        str: The rendered base summary prompt.
    """
    template = env.get_template("base_summary.jinja")
    return template.render(
        user_query=user_query,
        total_chunks=total_chunks,
        chunk_range_start=chunk_range_start,
        chunk_range_end=chunk_range_end,
        chunk_content=chunk_content
    )


def get_recursive_summary_prompt(user_query, total_chunks, summary_level, summary_range_start, 
                                summary_range_end, total_summaries, chunk_range_start, 
                                chunk_range_end, summaries):
    """Render the recursive summary prompt template.
    
    Args:
        user_query (str): The user's original query about the document.
        total_chunks (int): The total number of chunks in the document.
        summary_level (int): The current level of summarization (1, 2, etc.).
        summary_range_start (int): The starting summary number in this batch.
        summary_range_end (int): The ending summary number in this batch.
        total_summaries (int): The total number of summaries at this level.
        chunk_range_start (int): The starting chunk number covered by these summaries.
        chunk_range_end (int): The ending chunk number covered by these summaries.
        summaries (list): A list of summary dictionaries, each containing:
            - start_chunk (int): The starting chunk number for this summary.
            - end_chunk (int): The ending chunk number for this summary.
            - content (str): The content of the summary.
        
    Returns:
        str: The rendered recursive summary prompt.
    """
    template = env.get_template("recursive_summary.jinja")
    return template.render(
        user_query=user_query,
        total_chunks=total_chunks,
        summary_level=summary_level,
        summary_range_start=summary_range_start,
        summary_range_end=summary_range_end,
        total_summaries=total_summaries,
        chunk_range_start=chunk_range_start,
        chunk_range_end=chunk_range_end,
        summaries=summaries
    )


def get_final_answer_prompt(user_query, total_chunks, total_summary_levels, final_summary):
    """Render the final answer prompt template.
    
    Args:
        user_query (str): The user's original query about the document.
        total_chunks (int): The total number of chunks in the document.
        total_summary_levels (int): The total number of summary levels used.
        final_summary (str): The final top-level summary of the entire document.
        
    Returns:
        str: The rendered final answer prompt.
    """
    template = env.get_template("final_answer.jinja")
    return template.render(
        user_query=user_query,
        total_chunks=total_chunks,
        total_summary_levels=total_summary_levels,
        final_summary=final_summary
    )