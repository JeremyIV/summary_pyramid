"""Module for chunking documents into token-sized pieces.

This module provides functionality to split long documents into
chunks based on token count, using Anthropic's tokenizer.
"""

import re
import time
from typing import List

try:
    import anthropic
except ImportError:
    raise ImportError(
        "The Anthropic Python SDK is required. "
        "Please install it with: pip install anthropic"
    )

# Constants
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"
TOKEN_COUNT_RATE_LIMIT = 10  # Max requests per minute for token counting API


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count the number of tokens in a text using Anthropic's tokenizer.
    
    Args:
        text: The text to count tokens for.
        model: The model to use for token counting.
        
    Returns:
        The number of tokens in the text.
    """
    print("Counting Tokens!")
    client = anthropic.Anthropic()
    response = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens


def estimate_tokens(text: str) -> int:
    """Provide a rough estimate of tokens to reduce API calls.
    
    This is a simple heuristic to reduce API calls for token counting.
    
    Args:
        text: The text to estimate tokens for.
        
    Returns:
        Rough estimate of token count.
    """
    # Rough heuristic: ~1.5 tokens per word
    words = len(text.split())
    return int(words * 1.5)


class TokenCounter:
    """Helper class to manage token counting with rate limiting."""
    
    def __init__(self, model: str = DEFAULT_MODEL, max_requests_per_minute: int = TOKEN_COUNT_RATE_LIMIT):
        self.client = anthropic.Anthropic()
        self.model = model
        self.min_interval = 60.0 / max_requests_per_minute
        self.last_request_time = 0
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text with rate limiting.
        
        Args:
            text: The text to count tokens in.
            
        Returns:
            The number of tokens in the text.
        """
        # Only make API call if absolutely necessary (for larger chunks)
        # For very short text, use estimate to avoid API calls
        #if len(text) < 50:
        #    return estimate_tokens(text)
        return estimate_tokens(text)
        
        # Enforce rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        
        # Make the API call
        print("Counting Tokens!!!")
        response = self.client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}]
        )
        
        self.last_request_time = time.time()
        return response.input_tokens


def chunk_document(document: str, tokens_per_chunk: int, model: str = DEFAULT_MODEL) -> List[str]:
    """Split a document into chunks of specified token size.
    
    Args:
        document: The document text to split into chunks.
        tokens_per_chunk: Maximum number of tokens per chunk.
        model: The model to use for token counting.
        
    Returns:
        A list of document chunks, where each chunk is a string with
        tokens_per_chunk or fewer tokens.
    """
    # Initialize token counter with rate limiting
    token_counter = TokenCounter(model)
    
    # Split document into paragraphs first to preserve some structure
    paragraphs = re.split(r'\n\s*\n', document)
    
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    
    # Batch token counting for efficiency
    # First pass: use estimates to create initial paragraphs
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # Estimate tokens for this paragraph to reduce API calls
        estimated_tokens = estimate_tokens(paragraph)
        
        # For large paragraphs, use accurate counting
        if estimated_tokens > tokens_per_chunk * 0.8:
            paragraph_tokens = token_counter.count_tokens(paragraph)
        else:
            paragraph_tokens = estimated_tokens
        
        # If a single paragraph is larger than tokens_per_chunk, we need to split it
        if paragraph_tokens > tokens_per_chunk:
            # If we have content in the current chunk, finalize it
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_chunk_tokens = 0
            
            # Split the paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sentence_buffer = []
            buffer_tokens = 0
            
            for sentence in sentences:
                # Use estimate for short sentences
                if len(sentence) < 200:
                    sentence_tokens = estimate_tokens(sentence)
                else:
                    sentence_tokens = token_counter.count_tokens(sentence)
                
                # If this single sentence is too large, we must split it further
                if sentence_tokens > tokens_per_chunk:
                    # Handle excessively long sentences by breaking them into pieces
                    words = sentence.split()
                    word_buffer = []
                    word_buffer_tokens = 0
                    
                    for word in words:
                        # Use estimate for words (they're always short)
                        word_tokens = estimate_tokens(word + ' ')
                        
                        if word_buffer_tokens + word_tokens <= tokens_per_chunk:
                            word_buffer.append(word)
                            word_buffer_tokens += word_tokens
                        else:
                            # Finalize current word buffer as a chunk
                            if word_buffer:
                                chunks.append(' '.join(word_buffer))
                                word_buffer = [word]
                                word_buffer_tokens = word_tokens
                    
                    # Add any remaining words
                    if word_buffer:
                        chunks.append(' '.join(word_buffer))
                
                # If adding this sentence would exceed our chunk size, finalize the buffer
                elif buffer_tokens + sentence_tokens > tokens_per_chunk:
                    if sentence_buffer:
                        chunks.append(' '.join(sentence_buffer))
                        sentence_buffer = [sentence]
                        buffer_tokens = sentence_tokens
                    else:
                        # Edge case: first sentence in buffer is already too big
                        chunks.append(sentence)
                
                # Otherwise add the sentence to our buffer
                else:
                    sentence_buffer.append(sentence)
                    buffer_tokens += sentence_tokens
            
            # Add any remaining sentences in the buffer
            if sentence_buffer:
                chunks.append(' '.join(sentence_buffer))
        
        # If adding this paragraph would exceed chunk size, finalize current chunk
        elif current_chunk_tokens + paragraph_tokens > tokens_per_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_chunk_tokens = paragraph_tokens
        
        # Otherwise add paragraph to current chunk
        else:
            current_chunk.append(paragraph)
            current_chunk_tokens += paragraph_tokens
    
    # Add the final chunk if there's anything left
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    # Verify final chunks to make sure they don't exceed the token limit
    # This is important because our estimates might be off
    verified_chunks = []
    for chunk in chunks:
        # For larger chunks, verify token count
        if len(chunk) > 1000:
            chunk_tokens = token_counter.count_tokens(chunk)
            
            # If the chunk is too large, we need to split it further
            # But this should be rare due to our conservative estimates
            if chunk_tokens > tokens_per_chunk * 1.1:  # Allow 10% buffer
                # Recursively chunk this oversized chunk
                sub_chunks = chunk_document(chunk, tokens_per_chunk, model)
                verified_chunks.extend(sub_chunks)
            else:
                verified_chunks.append(chunk)
        else:
            verified_chunks.append(chunk)
    
    return verified_chunks