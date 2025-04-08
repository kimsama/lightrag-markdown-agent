#!/usr/bin/env python
"""
Setup environment variables for LightRAG Markdown Agent.
This script helps users set up their .env file with the necessary environment variables.
"""

import os
import sys
from pathlib import Path

def setup_env():
    """Set up the .env file with environment variables."""
    # Check if .env file exists
    env_file = Path('.env')
    if env_file.exists():
        print("Warning: .env file already exists.")
        overwrite = input("Do you want to overwrite it? (y/n): ").lower()
        if overwrite != 'y':
            print("Setup aborted.")
            return
    
    # Get OpenAI API key
    api_key = input("Enter your OpenAI API key: ")
    if not api_key:
        print("Error: OpenAI API key is required.")
        return
    
    # Get optional settings
    working_dir = input("Enter working directory (default: ./my_rag_data): ") or "./my_rag_data"
    output_dir = input("Enter output directory (default: ./output_images): ") or "./output_images"
    model_name = input("Enter model name (default: gpt-3.5-turbo): ") or "gpt-3.5-turbo"
    top_k = input("Enter top-k value (default: 60): ") or "60"
    max_image_size = input("Enter max image size as width,height (default: 800,800): ") or "800,800"
    image_quality = input("Enter image quality (default: 85): ") or "85"
    image_format = input("Enter image format (default: JPEG): ") or "JPEG"
    
    # Write to .env file
    with open(env_file, 'w') as f:
        f.write("# LightRAG Markdown Agent Environment Variables\n\n")
        f.write("# OpenAI API Key (Required)\n")
        f.write(f"OPENAI_API_KEY={api_key}\n\n")
        f.write("# Working Directory (Optional)\n")
        f.write(f"WORKING_DIR={working_dir}\n\n")
        f.write("# Output Directory (Optional)\n")
        f.write(f"OUTPUT_DIR={output_dir}\n\n")
        f.write("# Model Configuration (Optional)\n")
        f.write(f"MODEL_NAME={model_name}\n\n")
        f.write("# Top-K Configuration (Optional)\n")
        f.write(f"TOP_K={top_k}\n\n")
        f.write("# Image Processing Configuration (Optional)\n")
        f.write(f"MAX_IMAGE_SIZE={max_image_size}\n")
        f.write(f"IMAGE_QUALITY={image_quality}\n")
        f.write(f"IMAGE_FORMAT={image_format}\n")
    
    print(f"\nEnvironment variables have been set up in {env_file}.")
    print("You can now run the LightRAG Markdown Agent.")

if __name__ == "__main__":
    setup_env() 