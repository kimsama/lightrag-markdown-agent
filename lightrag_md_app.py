import os
import argparse
from typing import Optional
from pathlib import Path
from markdown_processor import MarkdownRAG
from image_processor import ImageProcessor
from citation_handler import CitationHandler
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description='LightRAG Markdown Processing and Querying')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process markdown files')
    process_parser.add_argument('--input', required=True, help='Input markdown file or directory')
    process_parser.add_argument('--working-dir', help='Working directory for RAG data')
    process_parser.add_argument('--api-key', help='OpenAI API key')
    process_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('--query', required=True, help='Query string')
    query_parser.add_argument('--mode', choices=['mix', 'local', 'global', 'hybrid', 'naive'],
                            default='mix', help='Query mode')
    query_parser.add_argument('--top-k', type=int, help='Number of top items to retrieve')
    query_parser.add_argument('--working-dir', help='Working directory with RAG data')
    query_parser.add_argument('--output', help='Directory to save images')
    query_parser.add_argument('--api-key', help='OpenAI API key')
    query_parser.add_argument('--metadata', help='Path to image metadata file')
    query_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    return parser

def process_markdown(args: argparse.Namespace) -> None:
    """Process markdown files.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    input_path = Path(args.input)
    
    # Use working directory from args or environment variable
    working_dir = Path(args.working_dir) if args.working_dir else Path(os.getenv('WORKING_DIR', './my_rag_data'))
    
    # Create working directory if it doesn't exist
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Set API key if provided
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or provide it with the --api-key argument.")
        print("You can create a .env file based on .env.example")
    
    # Initialize components
    image_processor = ImageProcessor(
        max_size=tuple(map(int, os.getenv('MAX_IMAGE_SIZE', '800,800').split(','))),
        quality=int(os.getenv('IMAGE_QUALITY', '85')),
        format=os.getenv('IMAGE_FORMAT', 'JPEG')
    )
    citation_handler = CitationHandler(working_dir=working_dir, verbose=args.verbose)
    rag = MarkdownRAG(
        working_dir=working_dir,
        image_processor=image_processor,
        citation_handler=citation_handler,
        model_name=os.getenv('MODEL_NAME', 'gpt-3.5-turbo'),
        top_k=int(os.getenv('TOP_K', '60')),
        verbose=args.verbose
    )
    
    # Process input
    if input_path.is_file():
        print(f"Processing file: {input_path}")
        rag.process_markdown_file(input_path)
    elif input_path.is_dir():
        print(f"Processing directory: {input_path}")
        rag.process_directory(input_path)
    else:
        print(f"Error: Input path does not exist: {input_path}")
        return
        
    print("Processing complete!")

def query_rag(args: argparse.Namespace) -> None:
    """Query the RAG system.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Use working directory from args or environment variable
    working_dir = Path(args.working_dir) if args.working_dir else Path(os.getenv('WORKING_DIR', './my_rag_data'))
    
    if not working_dir.exists():
        print(f"Error: Working directory does not exist: {working_dir}")
        return
    
    # Set API key if provided
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
        
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or provide it with the --api-key argument.")
        print("You can create a .env file based on .env.example")
        
    # Initialize components
    image_processor = ImageProcessor(
        max_size=tuple(map(int, os.getenv('MAX_IMAGE_SIZE', '800,800').split(','))),
        quality=int(os.getenv('IMAGE_QUALITY', '85')),
        format=os.getenv('IMAGE_FORMAT', 'JPEG')
    )
    citation_handler = CitationHandler(working_dir=working_dir, verbose=args.verbose)
    rag = MarkdownRAG(
        working_dir=working_dir,
        image_processor=image_processor,
        citation_handler=citation_handler,
        model_name=os.getenv('MODEL_NAME', 'gpt-3.5-turbo'),
        top_k=int(os.getenv('TOP_K', '60')),
        verbose=args.verbose
    )
    
    # Create output directory if specified
    output_dir = Path(args.output) if args.output else Path(os.getenv('OUTPUT_DIR', './output_images'))
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # Execute query
    response = rag.query(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k if args.top_k else int(os.getenv('TOP_K', '60'))
    )
    
    # Print response
    print("\nResponse:")
    print(response.text)
    
    if response.citations:
        print("\nCitations:")
        print(response.citations)
        
    if response.images:
        print("\nImages:")
        for img in response.images:
            print(f"- {img.get('alt_text', 'unknown')}: {img.get('path', 'unknown')}")
            
        # Save images to output directory if specified
        if output_dir:
            print(f"\nSaving images to {output_dir}")
            for img in response.images:
                img_id = img.get('img_id', 'unknown')
                img_url = img.get('path', '')
                
                if img_url:
                    try:
                        # Set a proper User-Agent header
                        headers = {
                            'User-Agent': 'LightRAG-Markdown-Agent/1.0 (https://github.com/yourusername/lightrag-markdown-agent; your-email@example.com)'
                        }
                        
                        # Download image
                        response = requests.get(img_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        
                        # Save image
                        img_path = output_dir / f"{img_id}.png"
                        with open(img_path, 'wb') as f:
                            f.write(response.content)
                        print(f"  Saved: {img_path}")
                    except Exception as e:
                        print(f"  Error saving image {img_id}: {str(e)}")

def main() -> None:
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command == 'process':
        process_markdown(args)
    elif args.command == 'query':
        query_rag(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 