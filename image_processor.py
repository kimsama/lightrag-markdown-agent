import base64
import os
import re
import requests
from io import BytesIO
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import io
import uuid
from pathlib import Path
import hashlib
from pydantic import BaseModel, Field

class ImageData(BaseModel):
    """Model for storing processed image data."""
    hash: str = Field(..., description="Unique hash of the image")
    path: str = Field(..., description="Original image path")
    alt_text: str = Field("", description="Image alt text")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    format: str = Field(..., description="Image format")
    base64_data: str = Field(..., description="Base64 encoded image data")
    metadata: Dict = Field(default_factory=dict, description="Additional image metadata")

class ImageProcessor:
    """Processor for handling images in markdown documents."""
    
    def __init__(
        self,
        max_size: Tuple[int, int] = (800, 800),
        quality: int = 85,
        format: str = "JPEG"
    ):
        """Initialize the image processor.
        
        Args:
            max_size (Tuple[int, int]): Maximum dimensions for processed images
            quality (int): JPEG quality (1-100)
            format (str): Output image format
        """
        self.max_size = max_size
        self.quality = quality
        self.format = format
        
    def _calculate_hash(self, image_data: bytes) -> str:
        """Calculate a unique hash for an image.
        
        Args:
            image_data (bytes): Raw image data
            
        Returns:
            str: Image hash
        """
        return hashlib.sha256(image_data).hexdigest()
        
    def _resize_image(
        self,
        image: Image.Image,
        max_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize an image while maintaining aspect ratio.
        
        Args:
            image (Image.Image): Image to resize
            max_size (Tuple[int, int]): Maximum dimensions
            
        Returns:
            Image.Image: Resized image
        """
        width, height = image.size
        max_width, max_height = max_size
        
        # Calculate scaling factor
        width_ratio = max_width / width
        height_ratio = max_height / height
        scale = min(width_ratio, height_ratio)
        
        # Only resize if needed
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        return image
        
    def _process_image_data(
        self,
        image_data: bytes,
        alt_text: str = "",
        metadata: Optional[Dict] = None
    ) -> Optional[ImageData]:
        """Process image data into a standardized format.
        
        Args:
            image_data (bytes): Raw image data
            alt_text (str): Image alt text
            metadata (Optional[Dict]): Additional metadata
            
        Returns:
            Optional[ImageData]: Processed image data if successful
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
                
            # Resize if needed
            image = self._resize_image(image, self.max_size)
            
            # Save to bytes
            output = io.BytesIO()
            image.save(
                output,
                format=self.format,
                quality=self.quality,
                optimize=True
            )
            processed_data = output.getvalue()
            
            # Calculate hash
            image_hash = self._calculate_hash(processed_data)
            
            # Create image data
            return ImageData(
                hash=image_hash,
                path="",  # Will be set by caller
                alt_text=alt_text,
                width=image.width,
                height=image.height,
                format=self.format,
                base64_data=base64.b64encode(processed_data).decode(),
                metadata=metadata or {}
            )
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
            
    def process_image(
        self,
        image_path: Union[str, Path],
        alt_text: str = "",
        metadata: Optional[Dict] = None
    ) -> Optional[ImageData]:
        """Process an image file.
        
        Args:
            image_path (Union[str, Path]): Path to image file
            alt_text (str): Image alt text
            metadata (Optional[Dict]): Additional metadata
            
        Returns:
            Optional[ImageData]: Processed image data if successful
        """
        try:
            # Read image file
            with open(image_path, 'rb') as f:
                image_data = f.read()
                
            # Process image
            image_data = self._process_image_data(
                image_data,
                alt_text=alt_text,
                metadata=metadata
            )
            
            if image_data:
                # Set original path
                image_data.path = str(image_path)
                
            return image_data
            
        except Exception as e:
            print(f"Error reading image {image_path}: {str(e)}")
            return None
            
    def process_base64(
        self,
        base64_data: str,
        alt_text: str = "",
        metadata: Optional[Dict] = None
    ) -> Optional[ImageData]:
        """Process a base64 encoded image.
        
        Args:
            base64_data (str): Base64 encoded image data
            alt_text (str): Image alt text
            metadata (Optional[Dict]): Additional metadata
            
        Returns:
            Optional[ImageData]: Processed image data if successful
        """
        try:
            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            
            # Process image
            return self._process_image_data(
                image_data,
                alt_text=alt_text,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error processing base64 image: {str(e)}")
            return None
            
    def save_image(
        self,
        image_data: ImageData,
        output_path: Union[str, Path]
    ) -> bool:
        """Save processed image data to a file.
        
        Args:
            image_data (ImageData): Processed image data
            output_path (Union[str, Path]): Output file path
            
        Returns:
            bool: True if successful
        """
        try:
            # Decode base64 data
            image_bytes = base64.b64decode(image_data.base64_data)
            
            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
                
            return True
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False

    def extract_image_urls(self, markdown_content: str) -> List[Tuple[str, str, str]]:
        """
        Extract image URLs from markdown content.
        
        Args:
            markdown_content: The markdown content to parse
            
        Returns:
            List of tuples containing (alt_text, image_url, image_type)
        """
        # Match both ![alt](url) and <img src="url" alt="alt"> formats
        img_pattern = r'!\[(.*?)\]\((.*?)\)|<img[^>]*src=[\'"](.*?)[\'"][^>]*alt=[\'"](.*?)[\'"][^>]*>'
        
        matches = re.findall(img_pattern, markdown_content)
        image_data = []
        
        for match in matches:
            if match[0]:  # ![alt](url) format
                alt_text, url = match[0], match[1]
            else:  # <img> tag format
                url, alt_text = match[2], match[3]
                
            # Determine image type from URL or content
            img_type = self._get_image_type(url)
            image_data.append((alt_text, url, img_type))
            
        return image_data
    
    def _get_image_type(self, url: str) -> str:
        """
        Determine the image type from the URL.
        
        Args:
            url: The image URL
            
        Returns:
            The image type (e.g., 'jpg', 'png')
        """
        # Try to get from file extension
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        if path.endswith('.jpg') or path.endswith('.jpeg'):
            return 'jpg'
        elif path.endswith('.png'):
            return 'png'
        elif path.endswith('.gif'):
            return 'gif'
        elif path.endswith('.webp'):
            return 'webp'
        elif path.endswith('.svg'):
            return 'svg'
        else:
            # Default to jpg if unknown
            return 'jpg'
    
    def download_image(self, url: str) -> Optional[bytes]:
        """
        Download an image from a URL.
        
        Args:
            url: The image URL
            
        Returns:
            The image data as bytes, or None if download failed
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None
    
    def encode_image_to_base64(self, image_data: bytes) -> str:
        """
        Encode image data as base64.
        
        Args:
            image_data: The image data as bytes
            
        Returns:
            The base64 encoded image
        """
        return base64.b64encode(image_data).decode('utf-8')
    
    def process_markdown_images(self, content: str) -> Tuple[str, Dict[str, Dict]]:
        """Process images in markdown content.
        
        Args:
            content (str): Markdown content
            
        Returns:
            Tuple[str, Dict[str, Dict]]: Processed content and image data
        """
        # Find all image references in markdown
        img_pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = re.finditer(img_pattern, content)
        
        processed_content = content
        image_data = {}
        
        for match in matches:
            alt_text = match.group(1)
            image_path = match.group(2)
            
            # Generate unique ID for the image
            img_id = f"img_{uuid.uuid4().hex[:8]}"
            
            # Process the image
            img_data = self._process_image(image_path, alt_text)
            if img_data:
                image_data[img_id] = img_data
                
                # Replace image reference with placeholder
                placeholder = f"![{alt_text}](#{img_id})"
                processed_content = processed_content.replace(match.group(0), placeholder)
                
        return processed_content, image_data
        
    def _process_image(self, image_path: str, alt_text: str) -> Optional[Dict]:
        """Process a single image.
        
        Args:
            image_path (str): Path to the image
            alt_text (str): Alt text for the image
            
        Returns:
            Optional[Dict]: Image data if successful
        """
        try:
            # Handle different image path formats
            if image_path.startswith('data:image'):
                # Handle base64 encoded images
                img_data = self._process_base64_image(image_path)
            else:
                # Handle file paths
                img_data = self._process_file_image(image_path)
                
            if img_data:
                img_data['alt_text'] = alt_text
                return img_data
                
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
            
    def _process_base64_image(self, base64_str: str) -> Optional[Dict]:
        """Process a base64 encoded image.
        
        Args:
            base64_str (str): Base64 encoded image string
            
        Returns:
            Optional[Dict]: Image data if successful
        """
        try:
            # Extract the base64 data
            header, encoded = base64_str.split(",", 1)
            image_data = base64.b64decode(encoded)
            
            # Generate filename
            filename = f"img_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(self.image_dir, filename)
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(image_data)
                
            return {
                'filename': filename,
                'path': filepath,
                'base64': encoded
            }
            
        except Exception as e:
            print(f"Error processing base64 image: {str(e)}")
            return None
            
    def _process_file_image(self, image_path: str) -> Optional[Dict]:
        """Process an image file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[Dict]: Image data if successful
        """
        try:
            # Open and process the image
            with Image.open(image_path) as img:
                # Convert to PNG format
                if img.format != 'PNG':
                    img = img.convert('RGBA')
                    
                # Generate filename
                filename = f"img_{uuid.uuid4().hex[:8]}.png"
                filepath = os.path.join(self.image_dir, filename)
                
                # Save the image
                img.save(filepath, 'PNG')
                
                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return {
                    'filename': filename,
                    'path': filepath,
                    'base64': img_str
                }
                
        except Exception as e:
            print(f"Error processing file image {image_path}: {str(e)}")
            return None
            
    def get_image_data(self, img_id: str) -> Optional[Dict]:
        """Get image data by ID.
        
        Args:
            img_id (str): Image ID
            
        Returns:
            Optional[Dict]: Image data if found
        """
        return self.image_data.get(img_id)
        
    def get_image_hash(self, image_path: Union[str, Path]) -> Optional[str]:
        """Calculate the hash of an image file.
        
        Args:
            image_path (Union[str, Path]): Path to the image file
            
        Returns:
            Optional[str]: SHA-256 hash of the image or None if failed
        """
        try:
            with open(image_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None 