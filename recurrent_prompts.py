"""Functions for rendering recurrent prompt templates.

This module contains functions to render Jinja templates for the recurrent summarization approach,
which processes documents sequentially section by section.
"""

import os
from jinja2 import Environment, FileSystemLoader

# Set up Jinja environment
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recurrent_prompt_templates")
env = Environment(loader=FileSystemLoader(template_dir))


def get_system_prompt(context_window_size, tokens_per_selection=5000, summary_token_limit=2000):
    """Render the system prompt template.
    
    Args:
        context_window_size (int): The size of the AI model's context window in tokens.
        tokens_per_selection (int): The size of each content section in tokens.
        summary_token_limit (int): The maximum size for summaries in tokens.
        
    Returns:
        str: The rendered system prompt.
    """
    template = env.get_template("system_prompt.jinja")
    return template.render(
        context_window_size=context_window_size,
        tokens_per_selection=tokens_per_selection,
        summary_token_limit=summary_token_limit
    )


def get_base_summary_prompt(user_query, total_chunks, chunk_content, summary_token_limit=2000):
    """Render the initial summary prompt for the first section of the document.
    
    Args:
        user_query (str): The user's query about the document.
        total_chunks (int): The total number of chunks in the document.
        chunk_content (str): The content of the first section to summarize.
        summary_token_limit (int): The maximum size for summaries in tokens.
        
    Returns:
        str: The rendered prompt for the initial summary.
    """
    template = env.get_template("base_summary.jinja")
    return template.render(
        user_query=user_query,
        total_chunks=total_chunks,
        chunk_content=chunk_content,
        summary_token_limit=summary_token_limit
    )


def get_recursive_summary_prompt(user_query, total_chunks, current_chunk, 
                               chunks_processed, current_summary, new_chunk_content,
                               summary_token_limit=2000):
    """Render the recursive summary prompt that updates the running summary with new content.
    
    Args:
        user_query (str): The user's query about the document.
        total_chunks (int): The total number of chunks in the document.
        current_chunk (int): The current chunk number being processed.
        chunks_processed (int): Number of chunks processed so far including the current one.
        current_summary (str): The current running summary of previously processed chunks.
        new_chunk_content (str): The content of the new chunk to incorporate.
        summary_token_limit (int): The maximum size for summaries in tokens.
        
    Returns:
        str: The rendered prompt for updating the summary.
    """
    template = env.get_template("recursive_summary.jinja")
    return template.render(
        user_query=user_query,
        total_chunks=total_chunks,
        current_chunk=current_chunk,
        chunks_processed=chunks_processed,
        current_summary=current_summary,
        new_chunk_content=new_chunk_content,
        summary_token_limit=summary_token_limit
    )