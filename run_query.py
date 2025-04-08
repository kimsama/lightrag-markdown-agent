import os
import sys
from pathlib import Path
from markdown_processor import MarkdownRAG
from image_processor import ImageProcessor
from citation_handler import CitationHandler

def main():
    # Set up working directory
    working_dir = Path("./my_rag_data")
    output_dir = Path("./images")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    image_processor = ImageProcessor(
        max_size=(800, 800),
        quality=85,
        format="JPEG"
    )
    citation_handler = CitationHandler(working_dir=working_dir)
    rag = MarkdownRAG(
        working_dir=working_dir,
        image_processor=image_processor,
        citation_handler=citation_handler,
        model_name="gpt-3.5-turbo",
        top_k=60
    )
    
    # Execute query
    query = "Describe the planets in our solar system"
    response = rag.query(
        query=query,
        mode="mix"
    )
    
    # Print response
    print("\nResponse:")
    print(response.text)
    
    # Print detailed information about the response
    print("\nResponse Details:")
    print(f"Type: {type(response)}")
    print(f"Text: {response.text[:100]}...")
    print(f"Citations: {response.citations}")
    print(f"Images: {response.images}")
    
    if response.citations:
        print("\nCitations:")
        print(response.citations)
        
    if response.images:
        print("\nImages:")
        for img in response.images:
            print(f"- {img.get('id', 'unknown')}: {img.get('url', 'unknown')}")
            
        # Save images to output directory
        print(f"\nSaving images to {output_dir}")
        for img in response.images:
            img_id = img.get('id', 'unknown')
            img_url = img.get('url', '')
            img_base64 = img.get('base64', '')
            
            if img_base64:
                # Save base64 image
                try:
                    import base64
                    from PIL import Image
                    from io import BytesIO
                    
                    # Decode base64 image
                    img_data = base64.b64decode(img_base64)
                    img_obj = Image.open(BytesIO(img_data))
                    
                    # Save image
                    img_path = output_dir / f"{img_id}.png"
                    img_obj.save(img_path)
                    print(f"  Saved: {img_path}")
                except Exception as e:
                    print(f"  Error saving image {img_id}: {str(e)}")
            elif img_url:
                # Download image from URL
                try:
                    import requests
                    from PIL import Image
                    from io import BytesIO
                    
                    # Set a proper User-Agent header
                    headers = {
                        'User-Agent': 'LightRAG-Markdown-Agent/1.0 (https://github.com/yourusername/lightrag-markdown-agent; your-email@example.com)'
                    }
                    
                    # Download image
                    response = requests.get(img_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    # Save image
                    img_obj = Image.open(BytesIO(response.content))
                    img_path = output_dir / f"{img_id}.png"
                    img_obj.save(img_path)
                    print(f"  Downloaded and saved: {img_path}")
                except Exception as e:
                    print(f"  Error downloading image {img_id}: {str(e)}")
    else:
        print("\nNo images found in the response.")

if __name__ == "__main__":
    main() 