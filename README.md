# Long Document QA: Convolutional vs. Recurrent Approaches

This repository contains tools for answering questions about very long documents that exceed an AI model's context window. It implements two different approaches to compare:

## Summary Pyramid: "Convolutional Reading"

Summary Pyramid uses a hierarchical, recursive summarization approach to distill information from large documents while preserving information relevant to a specific query.

The way Summary Pyramid aggregates information from across a large document is very similar to the way a convolutional neural network aggregates information from across a sequence. A sliding window summarizes different sections of the document, and then another sliding window summarizes those summaries, etc, exponentially reducing the total amount of text with each pass until we have distilled the document down to an amount of information that fits in the language model's context window.

## Summary Rollup: "Recurrent Reading"

Summary Rollup uses a sequential, recurrent summarization approach that processes the document chunk by chunk, maintaining and updating a running summary that incorporates information from each new section as it's processed.

These approaches might actually improve performance even if the entire document fits in the model's context window, as they provide more structured information processing than simply loading the entire document at once.

Both approaches can be compared against simply throwing everything in the context window when the document is small enough.

## Requirements

- Python 3.7+
- Anthropic API key (set as environment variable `ANTHROPIC_API_KEY`)

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install anthropic jinja2
```

3. Set up your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

The repository offers two different approaches for answering questions about long documents:

#### 1. Summary Pyramid (Convolutional Approach)

Run the pyramid tool from the command line:

```bash
python summary_pyramid.py --document path/to/your/document.txt --query "Your question about the document?"
```

This will create a summary pyramid and generate an answer to your query, with all outputs stored in the `pyramid_output` directory.

#### 2. Summary Rollup (Recurrent Approach)

Run the rollup tool from the command line:

```bash
python summary_rollup.py --document path/to/your/document.txt --query "Your question about the document?"
```

This will process the document sequentially, building a rolling summary, and generate an answer to your query, with all outputs stored in the `rollup_output` directory.

### Advanced Options

Both tools offer several options to customize their behavior:

```bash
python summary_pyramid.py --help
python summary_rollup.py --help
```

#### Summary Pyramid Parameters

- `--document`: Path to the document file (default: documents/mobydick.txt)
- `--query`: Path to query file or direct query string (default: queries/query.txt)
- `--tokens-per-chunk`: Size to chunk the document into (default: 1000)
- `--tokens-per-selection`: Target size for content selections (default: 5000)
- `--window-size`: Window size for all summary levels (default: 5)
- `--stride`: Stride for all summary levels (default: 4)
- `--model`: Claude model to use (default: claude-3-7-sonnet-20250219)
- `--output-dir`: Directory to store outputs (default: pyramid_output)

#### Summary Rollup Parameters

- `--document`: Path to the document file (default: documents/mobydick.txt)
- `--query`: Path to query file or direct query string (default: queries/query.txt)
- `--tokens-per-chunk`: Size to chunk the document into (default: 1000)
- `--tokens-per-selection`: Target size for content selections (default: 5000)
- `--context-window`: Context window size in tokens (default: 100000)
- `--summary-token-limit`: Maximum token limit for summaries (default: 2000)
- `--answer-token-limit`: Maximum token limit for final answer (default: 4000)
- `--model`: Claude model to use (default: claude-3-7-sonnet-20250219)
- `--output-dir`: Directory to store outputs (default: rollup_output)
- `--clear-output`: Clear output directory if it exists

## Output Structure

### Summary Pyramid Output

The pyramid tool creates a hierarchical output directory:

```
pyramid_output/
├── final_answer.txt                # The final answer to your query
├── final_summary.txt               # The top-level summary of the document
├── pyramid_metadata.txt            # Metadata about the pyramid structure
├── level_1/                        # Base level summaries
│   ├── chunks_1-3.txt
│   ├── chunks_3-5.txt
│   └── ...
├── level_2/                        # Second level summaries
│   ├── chunks_1-8.txt
│   └── ...
└── level_n/                        # Top level with single summary
    └── chunks_1-total.txt
```

### Summary Rollup Output

The rollup tool creates a sequential output directory:

```
rollup_output/
├── final_answer.txt                # The final answer to your query
├── final_summary.txt               # The final comprehensive summary
├── rollup_metadata.txt             # Metadata about the rollup process
└── summaries/                      # All intermediate summaries
    ├── summary_stage_1_of_n.txt    # Initial summary of first chunk
    ├── summary_stage_2_of_n.txt    # Updated summary with chunks 1-2
    └── ...                         # Each stage builds on previous stages
```

## Examples

### Using Summary Pyramid (Convolutional Approach)

Here's a simple example using Moby Dick with the pyramid approach:

```bash
python summary_pyramid.py --document documents/mobydick.txt --query "What is the significance of the white whale in this novel?" --tokens-per-chunk 2000
```

### Using Summary Rollup (Recurrent Approach)

Here's the same example using the rollup approach:

```bash
python summary_rollup.py --document documents/mobydick.txt --query "What is the significance of the white whale in this novel?" --tokens-per-chunk 2000
```

## Project Structure

- `summary_pyramid.py`: Main module with core convolutional summarization functions
- `summary_rollup.py`: Alternative module using a recurrent summarization approach
- `chunk_document.py`: Functions for chunking documents by token count
- `prompts.py`: Template rendering functions for Claude prompts for the pyramid approach
- `recurrent_prompts.py`: Template rendering functions for the rollup approach
- `prompt_templates/`: Jinja templates for various pyramid prompt types
- `recurrent_prompt_templates/`: Jinja templates for rollup prompt types

## Limitations

### Summary Pyramid
- Relies on hierarchical summarization, which may lose some nuanced details
- Performance depends on the quality of the base-level summaries
- Processing large documents can be time-consuming and API-intensive
- May have higher API usage due to multiple summary levels

### Summary Rollup
- Sequential processing means errors can propagate through the entire summary
- Later sections might receive less attention as the summary grows
- May struggle with documents that reference information non-linearly
- Potentially slower for very large documents as each chunk must be processed sequentially

## License

This project is licensed under the MIT License - see the LICENSE file for details.