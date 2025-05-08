# Summary Pyramid: "convolutional reading"

Summary Pyramid is a tool for answering questions about very long documents that exceed an AI model's context window. It uses a hierarchical, recursive summarization approach to distill information from large documents while preserving information relevant to a specific query.

The way Summary Pyramid aggregates information from across a large document is very similar to the way a convolutional neural network aggregates information from across a sequence. A sliding window summarizes different sections of the document, and then another sliding window summarizes those summaries, etc, exponentially reducing the total amount of text with each pass until we have distilled the document down to an amount of information that fits in the language model's context window.

I suspect but haven't yet demonstrated that this kind of approach could actually improve performance even if the entire document fits in the model's context window. This is just a hunch though.

Things to compare against:
 - throwing everything in the context window
 - a different long-document reading scaffolding framework based on a recurrent architecture instead of a convolutional one.

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

Run the tool from the command line:

```bash
python summary_pyramid.py --document path/to/your/document.txt --query "Your question about the document?"
```

This will create a summary pyramid and generate an answer to your query, with all outputs stored in the `pyramid_output` directory.

### Advanced Options

The tool offers several options to customize its behavior:

```bash
python summary_pyramid.py --help
```

Key parameters:

- `--document`: Path to the document file (default: documents/mobydick.txt)
- `--query`: Path to query file or direct query string (default: queries/query.txt)
- `--tokens-per-chunk`: Size to chunk the document into (default: 1000)
- `--tokens-per-selection`: Target size for content selections (default: 5000)
- `--window-size`: Window size for all summary levels (default: 5)
- `--stride`: Stride for all summary levels (default: 4)
- `--model`: Claude model to use (default: claude-3-7-sonnet-20250219)
- `--output-dir`: Directory to store outputs (default: pyramid_output)

## Output Structure

The tool creates a structured output directory:

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

## Example

Here's a simple example using Moby Dick:

```bash
python summary_pyramid.py --document documents/mobydick.txt --query "What is the significance of the white whale in this novel?" --tokens-per-chunk 2000
```

## Project Structure

- `summary_pyramid.py`: Main module with core summarization functions
- `chunk_document.py`: Functions for chunking documents by token count
- `prompts.py`: Template rendering functions for Claude prompts
- `prompt_templates/`: Jinja templates for various prompt types

## Limitations

- Relies on hierarchical summarization, which may lose some nuanced details
- Performance depends on the quality of the base-level summaries
- Processing large documents can be time-consuming and API-intensive

## License

This project is licensed under the MIT License - see the LICENSE file for details.