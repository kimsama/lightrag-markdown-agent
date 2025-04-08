# LightRAG Markdown Agent

A powerful RAG (Retrieval-Augmented Generation) system for processing and querying markdown documents with integrated image support.

## Features

- Process markdown files with text, tables, and images
- Automatic image encoding and storage
- Multiple query modes (mix, local, global, hybrid, naive)
- Citation tracking and bibliography generation
- Command-line interface and Streamlit web UI
- Support for both single file and directory processing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lightrag-markdown-agent.git
cd lightrag-markdown-agent

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

Process a single markdown file:
```bash
python lightrag_md_app.py process --input example.md --working-dir ./my_rag_data
```

Process all markdown files in a directory:
```bash
python lightrag_md_app.py process --input ./documents/ --working-dir ./my_rag_data
```

Query the system:
```bash
python lightrag_md_app.py query --query "Describe the planets in our solar system" --mode mix --working-dir ./my_rag_data --output ./images
```

### Web Interface

Launch the Streamlit web interface:
```bash
streamlit run streamlit_app.py
```

## Project Structure

```
lightrag-markdown-agent/
├── markdown_processor.py      # Core MarkdownRAG class
├── image_processor.py         # Image handling module
├── citation_handler.py        # Citation handling module
├── lightrag_md_app.py         # Command-line interface
├── streamlit_app.py           # Streamlit web interface
├── setup.py                   # Installation script
├── example.md                 # Example markdown file
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Configuration

The system can be configured using environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `WORKING_DIR`: Default working directory for RAG data
- `OUTPUT_DIR`: Default directory for saving images
- `MODEL_NAME`: Model to use for LightRAG (default: gpt-3.5-turbo)
- `TOP_K`: Number of top items to retrieve (default: 60)
- `MAX_IMAGE_SIZE`: Maximum image size as width,height (default: 800,800)
- `IMAGE_QUALITY`: Image quality for compression (default: 85)
- `IMAGE_FORMAT`: Image format for saving (default: JPEG)

### Setting Up Environment Variables

The easiest way to configure the application is to use a `.env` file:

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and set your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Optionally, customize other settings in the `.env` file.

Alternatively, you can use the setup script to create your `.env` file:

```bash
python setup_env.py
```

This script will guide you through setting up your environment variables interactively.

You can also set environment variables directly in your shell:

```bash
# On Linux/macOS
export OPENAI_API_KEY=your_openai_api_key_here

# On Windows (PowerShell)
$env:OPENAI_API_KEY="your_openai_api_key_here"

# On Windows (Command Prompt)
set OPENAI_API_KEY=your_openai_api_key_here
```

Alternatively, you can provide the API key directly when running commands:

```bash
python lightrag_md_app.py process --input example.md --working-dir ./my_rag_data --api-key your_openai_api_key_here
```

## Query Modes

- `mix`: Balanced approach combining local and global context
- `local`: Focus on local context and relationships
- `global`: Consider global document structure
- `hybrid`: Adaptive approach based on query type
- `naive`: Simple keyword-based retrieval

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 