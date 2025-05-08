# Summary Pyramid

Summary Pyramid is a tool for answering questions about very long documents that exceed an AI model's context window. It uses a hierarchical, recursive summarization approach to distill information from large documents while preserving information relevant to a specific query.

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