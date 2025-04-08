import os
import re
import uuid
import asyncio
import requests
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from pydantic import BaseModel, Field
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import markdown
from bs4 import BeautifulSoup
import hashlib
import base64
from io import BytesIO
from PIL import Image

from image_processor import ImageProcessor, ImageData
from citation_handler import CitationHandler, Citation

class MarkdownSection(BaseModel):
    """Model for storing a section of markdown content."""
    id: str = Field(..., description="Unique section ID")
    content: str = Field(..., description="Raw markdown content")
    html: str = Field(..., description="HTML rendered content")
    images: List[ImageData] = Field(default_factory=list, description="Images in the section")
    citations: List[str] = Field(default_factory=list, description="Citation IDs in the section")
    metadata: Dict = Field(default_factory=dict, description="Additional section metadata")

class MarkdownDocument(BaseModel):
    """Model for storing a complete markdown document."""
    id: str = Field(..., description="Unique document ID")
    path: str = Field(..., description="Original file path")
    title: str = Field(..., description="Document title")
    sections: List[MarkdownSection] = Field(default_factory=list, description="Document sections")
    images: Dict[str, ImageData] = Field(default_factory=dict, description="All images in document")
    citations: Dict[str, Citation] = Field(default_factory=dict, description="All citations in document")
    metadata: Dict = Field(default_factory=dict, description="Additional document metadata")

class MarkdownContent(BaseModel):
    """Model for storing processed markdown content."""
    text: str = Field(..., description="The text content of the markdown")
    tables: List[List[List[str]]] = Field(default_factory=list, description="Extracted tables from markdown")
    images: List[Dict] = Field(default_factory=list, description="Image data with base64 encoding")
    citations: List[Dict] = Field(default_factory=list, description="Citation information")

class RAGResponse:
    """Model for storing the response from the RAG system."""
    def __init__(self, text: str, citations: List[str], images: List[Dict]):
        self.text = text
        self.citations = citations
        self.images = images

class MarkdownRAG:
    """
    Core class for processing markdown files with LightRAG.
    Integrates with LightRAG for knowledge graph creation and querying.
    """
    
    def __init__(
        self,
        working_dir: Union[str, Path],
        image_processor: Optional[ImageProcessor] = None,
        citation_handler: Optional[CitationHandler] = None,
        model_name: str = "gpt-3.5-turbo",
        top_k: int = 60,
        verbose: bool = False
    ):
        """Initialize the MarkdownRAG system.
        
        Args:
            working_dir (Union[str, Path]): Working directory for storing data
            image_processor (Optional[ImageProcessor]): Image processor instance
            citation_handler (Optional[CitationHandler]): Citation handler instance
            model_name (str): Name of the model to use for LightRAG
            top_k (int): Number of top items to retrieve
            verbose (bool): Whether to enable verbose logging
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Create necessary directories
        self.rag_dir = os.path.join(str(self.working_dir), 'rag_data')
        self.image_dir = os.path.join(str(self.working_dir), 'images')
        os.makedirs(self.rag_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize components
        self.image_processor = image_processor or ImageProcessor()
        self.citation_handler = citation_handler or CitationHandler(working_dir=working_dir, verbose=verbose)
        
        # Store top_k for later use
        self.top_k = top_k
        
        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY environment variable not set.")
            print("LightRAG will not be able to process documents without an API key.")
            print("Please set it using: export OPENAI_API_KEY=your_api_key")
        
        # Initialize LightRAG with a simpler graph storage implementation
        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
            llm_model_name=model_name,
            graph_storage="NetworkXStorage"
        )
        
        # Initialize LightRAG storages
        asyncio.run(self._initialize_storages())
    
    async def _initialize_storages(self):
        """Initialize the LightRAG storages asynchronously."""
        await initialize_pipeline_status()
        await self.rag.initialize_storages()
    
    def process_markdown_file(self, file_path: Union[str, Path]) -> None:
        """Process a markdown file and add it to the RAG system.
        
        Args:
            file_path (Union[str, Path]): Path to the markdown file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Process the markdown file
        content = self.process_markdown(file_path)
        
        # Create metadata for the document
        metadata = {
            "file_path": str(file_path),
            "title": file_path.stem,
            "tables": content.tables,
            "images": content.images,
            "citations": content.citations
        }
        
        # Add the content to the RAG system
        # LightRAG.insert() only accepts the document content as a string
        self.rag.insert(content.text)
        
        # Store metadata separately
        # The first parameter should be the document ID, and the second should be the file path
        doc_id = str(file_path)
        self.citation_handler.add_document(doc_id, str(file_path))
        
        # Store additional metadata if needed
        # We can add a method to store additional metadata in the CitationHandler
        
        print(f"Processed file: {file_path}")
    
    def process_markdown(self, file_path: Union[str, Path]) -> MarkdownContent:
        """Process a markdown file and extract its content.
        
        Args:
            file_path (Union[str, Path]): Path to the markdown file
            
        Returns:
            MarkdownContent: Processed markdown content
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Read the markdown file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract tables
        tables = []
        for table in re.finditer(r'\|.*\|.*\n\|[-\s|]*\n(\|.*\|.*\n)*', content):
            table_text = table.group(0)
            # Convert table to list of lists
            rows = [
                [cell.strip() for cell in row.split('|')[1:-1]]
                for row in table_text.strip().split('\n')
                if row.strip() and not all(c in '- |' for c in row)
            ]
            tables.append(rows)
            
        # Extract images
        images = []
        for match in re.finditer(r'!\[(.*?)\]\((.*?)\)', content):
            alt_text, image_path = match.groups()
            
            # Check if the image path is a URL
            if image_path.startswith(('http://', 'https://')):
                try:
                    # Download and process the image
                    import requests
                    from PIL import Image
                    from io import BytesIO
                    
                    # Set a proper User-Agent header
                    headers = {
                        'User-Agent': 'LightRAG-Markdown-Agent/1.0 (https://github.com/yourusername/lightrag-markdown-agent; your-email@example.com)'
                    }
                    
                    # Download image
                    response = requests.get(image_path, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    # Process the image
                    img_obj = Image.open(BytesIO(response.content))
                    
                    # Convert RGBA to RGB if needed
                    if img_obj.mode == 'RGBA':
                        background = Image.new('RGB', img_obj.size, (255, 255, 255))
                        background.paste(img_obj, mask=img_obj.split()[3])
                        img_obj = background
                    
                    # Save the image to a BytesIO object
                    img_buffer = BytesIO()
                    img_obj.save(img_buffer, format=self.image_processor.format)
                    img_buffer.seek(0)
                    
                    # Process the image data
                    img_info = self.image_processor._process_image_data(
                        img_buffer.read(),
                        alt_text=alt_text,
                        metadata={"url": image_path}
                    )
                    if img_info:
                        img_info.path = image_path  # Set the URL as the path
                        
                        # Save the processed image to disk
                        img_filename = f"{img_info.hash[:8]}.{self.image_processor.format.lower()}"
                        img_path = Path(self.image_dir) / img_filename
                        img_obj.save(img_path, format=self.image_processor.format, quality=self.image_processor.quality)
                        print(f"Saved image to: {img_path}")
                        
                        # Add local path to metadata
                        img_info_dict = img_info.dict()
                        img_info_dict['local_path'] = str(img_path)
                        images.append(img_info_dict)
                        
                        # Store in citation handler
                        self.citation_handler.add_image(
                            img_info.hash,
                            img_info_dict
                        )
                except Exception as e:
                    print(f"Error processing image URL {image_path}: {str(e)}")
            else:
                # Process local image file
                try:
                    img_info = self.image_processor.process_image(
                        image_path,
                        alt_text=alt_text
                    )
                    if img_info:
                        # Save the processed image to disk
                        img_filename = f"{img_info.hash[:8]}.{self.image_processor.format.lower()}"
                        img_path = Path(self.image_dir) / img_filename
                        
                        # Convert base64 to image and save
                        img_data = base64.b64decode(img_info.base64_data)
                        img_obj = Image.open(BytesIO(img_data))
                        img_obj.save(img_path, format=self.image_processor.format, quality=self.image_processor.quality)
                        print(f"Saved image to: {img_path}")
                        
                        # Add local path to metadata
                        img_info_dict = img_info.dict()
                        img_info_dict['local_path'] = str(img_path)
                        images.append(img_info_dict)
                        
                        # Store in citation handler
                        self.citation_handler.add_image(
                            img_info.hash,
                            img_info_dict
                        )
                except Exception as e:
                    print(f"Error processing image file {image_path}: {str(e)}")
            
        # Extract citations
        citations = []
        for match in re.finditer(r'\[([^\]]+)\]\(([^\)]+)\)', content):
            text, url = match.groups()
            if not text.startswith('!'):  # Skip image citations
                citations.append({
                    'text': text,
                    'url': url
                })
                
        return MarkdownContent(
            text=content,
            tables=tables,
            images=images,
            citations=citations
        )
    
    def process_directory(self, dir_path: Union[str, Path]) -> List[MarkdownContent]:
        """Process all markdown files in a directory."""
        dir_path = Path(dir_path)
        contents = []
        
        for md_file in dir_path.glob('**/*.md'):
            content = self.process_markdown(md_file)
            contents.append(content)
            
        return contents
        
    def query(
        self,
        query: str,
        mode: str = "mix",
        top_k: Optional[int] = None
    ) -> RAGResponse:
        """Query the RAG system.
        
        Args:
            query (str): Query string
            mode (str): Query mode (mix, local, global, hybrid, naive)
            top_k (Optional[int]): Number of top items to retrieve
            
        Returns:
            RAGResponse: Response from the RAG system
        """
        # Use the top_k from the instance if not provided
        if top_k is None:
            top_k = self.top_k
            
        # Create a QueryParam object for the query mode
        query_param = QueryParam(mode=mode)
            
        # Query the RAG system
        response = self.rag.query(
            query,
            param=query_param
        )
        
        # Extract citations
        citations = self.citation_handler.find_citations(query, response)
        
        # Extract images
        images = self._extract_images_from_response(response)
        
        # Format response text with image references
        response_text = response
        if images:
            response_text += "\n\n## Images\n"
            for img in images:
                alt_text = img.get('alt_text', 'Image')
                path = img.get('path', '')
                if path:
                    response_text += f"\n![{alt_text}]({path})"
        
        return RAGResponse(
            text=response_text,
            citations=citations,
            images=images
        )
        
    def save_processed_content(self, content: MarkdownContent, output_path: Union[str, Path]):
        """Save processed content to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content.json(indent=2))
        
    def save_state(self):
        """Save the current state of the RAG system."""
        state = {
            'working_dir': str(self.working_dir),
            'rag_dir': self.rag_dir,
            'image_dir': self.image_dir
        }
        
        state_file = os.path.join(str(self.working_dir), 'state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self):
        """Load the state of the RAG system."""
        state_file = os.path.join(str(self.working_dir), 'state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.working_dir = Path(state['working_dir'])
                self.rag_dir = state['rag_dir']
                self.image_dir = state['image_dir']
        
        # Reinitialize LightRAG
        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
            llm_model_name=self.rag.llm_model_name
        )
        
        # Initialize LightRAG storages
        asyncio.run(self._initialize_storages())

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled.
        
        Args:
            message (str): Message to log
        """
        if self.verbose:
            print(message)
            
    def _extract_images_from_response(self, response: str) -> List[Dict]:
        """Extract images from a response string.
        
        Args:
            response (str): Response string
            
        Returns:
            List[Dict]: List of image information
        """
        self._log("\nDEBUG: Starting image extraction from response")
        images = []
        
        # Find image references in the response
        self._log("DEBUG: Searching for image references in response")
        for match in re.finditer(r'!\[(.*?)\]\((.*?)\)', response):
            alt_text, image_path = match.groups()
            self._log(f"DEBUG: Found image reference - Alt: {alt_text}, Path: {image_path}")
            
            # Check if the image is in our database
            img_info = self.citation_handler.find_image(image_path)
            if img_info:
                self._log(f"DEBUG: Image found in database: {img_info}")
                images.append(img_info)
            else:
                self._log(f"DEBUG: Image not found in database: {image_path}")
                
        # If no images were found in the response, try to find relevant images based on keywords
        if not images:
            self._log("DEBUG: No images found in response, trying keyword-based search")
            
            # Extract planet names from the response
            planet_names = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
            found_planets = []
            
            # First, look for exact planet names in the response
            for planet in planet_names:
                if planet in response.lower():
                    found_planets.append(planet)
                    self._log(f"DEBUG: Found planet name in response: {planet}")
            
            # If no planets found, extract general keywords
            if not found_planets:
                keywords = re.findall(r'\b\w{4,}\b', response.lower())
                keywords = [k for k in keywords if k not in ['the', 'and', 'that', 'this', 'with', 'from', 'have', 'what', 'when', 'where', 'which', 'there', 'their', 'they', 'them', 'then', 'than', 'this', 'those', 'these']]
                self._log(f"DEBUG: Extracted general keywords: {keywords[:10]}")
                
                # Add planet-specific keywords
                keywords.extend(planet_names)
            
            # Search for images using found planets or keywords
            search_terms = found_planets if found_planets else keywords[:10]
            self._log(f"DEBUG: Searching with terms: {search_terms}")
            
            for term in search_terms:
                self._log(f"DEBUG: Searching for images matching term: {term}")
                for img_id, img_info in self.citation_handler.images.items():
                    # Check if term matches alt_text (case-insensitive)
                    if term.lower() in img_info.get('alt_text', '').lower():
                        self._log(f"DEBUG: Found matching image for term '{term}': {img_info}")
                        if img_info not in images:
                            images.append(img_info)
                            break  # Only add one image per term
                            
        self._log(f"DEBUG: Final list of images: {images}")
        return images

class MarkdownProcessor:
    """Core processor for handling markdown documents with images and citations."""
    
    def __init__(
        self,
        working_dir: Union[str, Path],
        max_image_size: Tuple[int, int] = (800, 800)
    ):
        """Initialize the markdown processor.
        
        Args:
            working_dir (Union[str, Path]): Working directory for storing data
            max_image_size (Tuple[int, int]): Maximum dimensions for processed images
        """
        self.working_dir = Path(working_dir)
        self.image_processor = ImageProcessor(max_size=max_image_size)
        self.citation_handler = CitationHandler(working_dir)
        self.documents: Dict[str, MarkdownDocument] = {}
        self._load_documents()
        
    def _load_documents(self) -> None:
        """Load existing documents from disk."""
        docs_file = self.working_dir / "documents.json"
        if docs_file.exists():
            try:
                with open(docs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = {
                        k: MarkdownDocument(**v) for k, v in data.items()
                    }
            except Exception as e:
                print(f"Error loading documents: {str(e)}")
                
    def _save_documents(self) -> None:
        """Save documents to disk."""
        try:
            self.working_dir.mkdir(parents=True, exist_ok=True)
            docs_file = self.working_dir / "documents.json"
            with open(docs_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {k: v.dict() for k, v in self.documents.items()},
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        except Exception as e:
            print(f"Error saving documents: {str(e)}")
            
    def _generate_section_id(self, content: str) -> str:
        """Generate a unique ID for a section based on its content."""
        return f"section_{hashlib.sha256(content.encode()).hexdigest()[:8]}"
        
    def _extract_images(self, markdown_content: str) -> List[Tuple[str, str]]:
        """Extract image references from markdown content.
        
        Args:
            markdown_content (str): Markdown text to process
            
        Returns:
            List[Tuple[str, str]]: List of (alt_text, image_path) tuples
        """
        # Match both ![alt](path) and <img src="path" alt="alt"> formats
        pattern = r'!\[(.*?)\]\((.*?)\)|<img[^>]+src="([^"]+)"[^>]+alt="([^"]+)"'
        matches = re.finditer(pattern, markdown_content)
        return [(m.group(1) or m.group(4), m.group(2) or m.group(3)) for m in matches]
        
    def _process_section(
        self,
        content: str,
        doc_path: str,
        section_id: Optional[str] = None
    ) -> MarkdownSection:
        """Process a section of markdown content.
        
        Args:
            content (str): Markdown content to process
            doc_path (str): Path of the parent document
            section_id (Optional[str]): Optional section ID
            
        Returns:
            MarkdownSection: Processed section data
        """
        # Generate section ID if not provided
        if not section_id:
            section_id = self._generate_section_id(content)
            
        # Convert markdown to HTML
        html = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        
        # Process images
        images = []
        for alt_text, img_path in self._extract_images(content):
            # Resolve relative paths
            abs_path = str(Path(doc_path).parent / img_path)
            
            # Process image
            img_data = self.image_processor.process_image(abs_path)
            if img_data:
                images.append(img_data)
                
        # Extract and process citations
        citations = []
        for match in re.finditer(r'\[(.*?)\]\((.*?)\)', content):
            text, ref = match.groups()
            if ref.startswith('cite_'):
                citations.append(ref)
                
        return MarkdownSection(
            id=section_id,
            content=content,
            html=html,
            images=images,
            citations=citations
        )
        
    def process_document(
        self,
        file_path: Union[str, Path],
        split_sections: bool = True
    ) -> MarkdownDocument:
        """Process a markdown document.
        
        Args:
            file_path (Union[str, Path]): Path to markdown file
            split_sections (bool): Whether to split into sections
            
        Returns:
            MarkdownDocument: Processed document data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        # Read document content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Generate document ID
        doc_id = f"doc_{hashlib.sha256(str(file_path).encode()).hexdigest()[:8]}"
        
        # Split into sections if requested
        sections = []
        if split_sections:
            # Split on headers
            section_contents = re.split(r'(?m)^#{1,6}\s+', content)[1:]
            headers = re.findall(r'(?m)^#{1,6}\s+(.*?)$', content)
            
            for header, section_content in zip(headers, section_contents):
                section = self._process_section(
                    f"# {header}\n{section_content}",
                    str(file_path)
                )
                sections.append(section)
        else:
            # Process as single section
            section = self._process_section(content, str(file_path))
            sections.append(section)
            
        # Collect all images and citations
        images = {}
        citations = {}
        for section in sections:
            for img in section.images:
                images[img.hash] = img
            for cite_id in section.citations:
                citation = self.citation_handler.get_citation(cite_id)
                if citation:
                    citations[cite_id] = citation
                    
        # Create document
        doc = MarkdownDocument(
            id=doc_id,
            path=str(file_path),
            title=file_path.stem,
            sections=sections,
            images=images,
            citations=citations
        )
        
        # Store document
        self.documents[doc_id] = doc
        self._save_documents()
        
        return doc
        
    def get_document(self, doc_id: str) -> Optional[MarkdownDocument]:
        """Retrieve a document by ID.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            Optional[MarkdownDocument]: Document if found
        """
        return self.documents.get(doc_id)
        
    def search_documents(
        self,
        query: str,
        field: str = "content"
    ) -> List[Tuple[MarkdownDocument, float]]:
        """Search documents for matching content.
        
        Args:
            query (str): Search query
            field (str): Field to search (content, title, etc.)
            
        Returns:
            List[Tuple[MarkdownDocument, float]]: List of (document, score) tuples
        """
        results = []
        query = query.lower()
        
        for doc in self.documents.values():
            score = 0.0
            if field == "content":
                # Search in all sections
                for section in doc.sections:
                    if query in section.content.lower():
                        score += 1.0
            elif field == "title":
                if query in doc.title.lower():
                    score = 1.0
                    
            if score > 0:
                results.append((doc, score))
                
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
        
    def get_section_with_images(
        self,
        doc_id: str,
        section_id: str
    ) -> Optional[Tuple[MarkdownSection, List[ImageData]]]:
        """Get a section and its associated images.
        
        Args:
            doc_id (str): Document ID
            section_id (str): Section ID
            
        Returns:
            Optional[Tuple[MarkdownSection, List[ImageData]]]: Section and images if found
        """
        doc = self.get_document(doc_id)
        if not doc:
            return None
            
        for section in doc.sections:
            if section.id == section_id:
                return section, section.images
                
        return None 