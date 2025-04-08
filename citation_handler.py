import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path
import re

class DocumentCitation(BaseModel):
    """Model for document citations."""
    doc_id: str
    title: str
    path: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sections: List[str] = Field(default_factory=list)

class ImageCitation(BaseModel):
    """Model for image citations."""
    img_id: str
    alt_text: str
    path: str
    timestamp: datetime = Field(default_factory=datetime.now)
    doc_id: Optional[str] = None

class Citation(BaseModel):
    """Model for storing citation information."""
    id: str = Field(..., description="Unique citation ID")
    text: str = Field(..., description="Cited text content")
    source: str = Field(..., description="Source document or file")
    page: Optional[int] = Field(None, description="Page number if applicable")
    context: Optional[str] = Field(None, description="Additional context about the citation")
    metadata: Dict = Field(default_factory=dict, description="Additional citation metadata")

class CitationHandler:
    """Handles citation tracking and management for the RAG system."""
    
    def __init__(self, working_dir: Union[str, Path], verbose: bool = False):
        """Initialize the citation handler.
        
        Args:
            working_dir (Union[str, Path]): Working directory for storing citation data
            verbose (bool): Whether to enable verbose logging
        """
        self.working_dir = Path(working_dir)
        self.verbose = verbose
        self.citations: Dict[str, Citation] = {}
        self.citation_file = self.working_dir / "citations.json"
        self.documents: Dict[str, Dict] = {}
        self.images: Dict[str, Dict] = {}
        self.doc_citations: Dict[str, DocumentCitation] = {}
        self.img_citations: Dict[str, ImageCitation] = {}
        self._load_citations()
    
    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled.
        
        Args:
            message (str): Message to log
        """
        if self.verbose:
            print(message)
    
    def _load_citations(self) -> None:
        """Load existing citations from disk."""
        if self.citation_file.exists():
            try:
                with open(self.citation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Convert ISO format strings back to datetime objects
                    def parse_datetime(obj):
                        if isinstance(obj, str) and len(obj) >= 19 and obj[10] == 'T':
                            try:
                                return datetime.fromisoformat(obj)
                            except ValueError:
                                return obj
                        elif isinstance(obj, dict):
                            return {k: parse_datetime(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [parse_datetime(item) for item in obj]
                        else:
                            return obj
                    
                    # Load citations
                    citations_data = data.get('citations', {})
                    self.citations = {
                        k: Citation(**parse_datetime(v)) for k, v in citations_data.items()
                    }
                    
                    # Load documents
                    documents_data = data.get('documents', {})
                    self.documents = parse_datetime(documents_data)
                    self.doc_citations = {
                        doc_id: DocumentCitation(**parse_datetime(citation_data))
                        for doc_id, citation_data in documents_data.items()
                    }
                    
                    # Load images
                    images_data = data.get('images', {})
                    self.images = parse_datetime(images_data)
                    self.img_citations = {
                        img_id: ImageCitation(**parse_datetime(citation_data))
                        for img_id, citation_data in images_data.items()
                    }
            except Exception as e:
                print(f"Error loading citations: {str(e)}")
    
    def _save_citations(self) -> None:
        """Save citations to disk."""
        try:
            self.working_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert datetime objects to ISO format strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            # Convert citations to dictionaries and handle datetime serialization
            citations_dict = {k: convert_datetime(v.dict()) for k, v in self.citations.items()}
            documents_dict = {k: convert_datetime(v.dict()) for k, v in self.doc_citations.items()}
            images_dict = {k: convert_datetime(v.dict()) for k, v in self.img_citations.items()}
            
            data = {
                'citations': citations_dict,
                'documents': documents_dict,
                'images': images_dict
            }
            
            with open(self.citation_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving citations: {str(e)}")
    
    def add_document(self, doc_id: str, file_path: str):
        """Add a document to the citation system.
        
        Args:
            doc_id (str): Unique document identifier
            file_path (str): Path to the document file
        """
        self.documents[doc_id] = {
            'path': file_path,
            'added_at': datetime.now().isoformat(),
            'citations': []
        }
        self.doc_citations[doc_id] = DocumentCitation(
            doc_id=doc_id,
            title=doc_id,
            path=file_path
        )
        self._save_citations()
    
    def add_image(self, image_id: str, image_data: Dict):
        """Add an image to the citation system.
        
        Args:
            image_id (str): Unique image identifier
            image_data (Dict): Image metadata and data
        """
        self.images[image_id] = {
            'data': image_data,
            'added_at': datetime.now().isoformat(),
            'citations': []
        }
        self.img_citations[image_id] = ImageCitation(
            img_id=image_id,
            alt_text=image_data.get('alt_text', ''),
            path=image_data.get('path', ''),
            doc_id=image_data.get('doc_id')
        )
        self._save_citations()
    
    def add_citation(
        self,
        text: str,
        source: str,
        page: Optional[int] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a new citation to the system.
        
        Args:
            text (str): Cited text content
            source (str): Source document or file
            page (Optional[int]): Page number if applicable
            context (Optional[str]): Additional context
            metadata (Optional[Dict]): Additional metadata
            
        Returns:
            str: Citation ID
        """
        # Generate a unique ID based on content hash
        citation_id = f"cite_{hash(text + source) & 0xFFFFFFFF:08x}"
        
        # Create citation object
        citation = Citation(
            id=citation_id,
            text=text,
            source=source,
            page=page,
            context=context,
            metadata=metadata or {}
        )
        
        # Store citation
        self.citations[citation_id] = citation
        self._save_citations()
        
        return citation_id
    
    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Retrieve a citation by ID.
        
        Args:
            citation_id (str): Citation ID
            
        Returns:
            Optional[Citation]: Citation object if found
        """
        return self.citations.get(citation_id)
    
    def find_citations(
        self,
        text: str,
        source: Optional[str] = None
    ) -> List[Citation]:
        """Find citations matching the given text and optionally source.
        
        Args:
            text (str): Text to search for
            source (Optional[str]): Source to filter by
            
        Returns:
            List[Citation]: List of matching citations
        """
        matches = []
        for citation in self.citations.values():
            if text in citation.text and (source is None or citation.source == source):
                matches.append(citation)
        return matches
    
    def format_citation(self, citation_id: str, format: str = "text") -> Optional[str]:
        """Format a citation in the specified format.
        
        Args:
            citation_id (str): Citation ID
            format (str): Output format (text, html, markdown)
            
        Returns:
            Optional[str]: Formatted citation text
        """
        citation = self.get_citation(citation_id)
        if not citation:
            return None
            
        if format == "text":
            return f"[{citation.id}] {citation.text} (Source: {citation.source})"
        elif format == "html":
            return f'<cite id="{citation.id}">{citation.text}</cite> (Source: {citation.source})'
        elif format == "markdown":
            return f"[{citation.text}][{citation.id}]"
        else:
            return str(citation)
    
    def extract_citations_from_text(self, text: str) -> List[str]:
        """Extract citation IDs from text.
        
        Args:
            text (str): Text containing citation references
            
        Returns:
            List[str]: List of citation IDs found in the text
        """
        # Match citation patterns like [cite_12345678]
        pattern = r'\[cite_[0-9a-f]{8}\]'
        return re.findall(pattern, text)
    
    def get_citation_bibliography(
        self,
        citation_ids: List[str],
        format: str = "text"
    ) -> str:
        """Generate a bibliography for the given citation IDs.
        
        Args:
            citation_ids (List[str]): List of citation IDs
            format (str): Output format (text, html, markdown)
            
        Returns:
            str: Formatted bibliography
        """
        bibliography = []
        for i, citation_id in enumerate(citation_ids, 1):
            citation = self.get_citation(citation_id)
            if citation:
                if format == "text":
                    bibliography.append(
                        f"{i}. {citation.text}\n   Source: {citation.source}"
                    )
                elif format == "html":
                    bibliography.append(
                        f'<div class="citation">{i}. {citation.text}<br>'
                        f'Source: {citation.source}</div>'
                    )
                elif format == "markdown":
                    bibliography.append(
                        f"{i}. {citation.text}  \n   Source: {citation.source}"
                    )
                    
        return "\n\n".join(bibliography)
    
    def get_document_citations(self, doc_id: str) -> List[Dict]:
        """Get all citations for a document.
        
        Args:
            doc_id (str): Document identifier
            
        Returns:
            List[Dict]: List of citations
        """
        return self.documents.get(doc_id, {}).get('citations', [])
    
    def get_image_citations(self, image_id: str) -> List[Dict]:
        """Get all citations for an image.
        
        Args:
            image_id (str): Image identifier
            
        Returns:
            List[Dict]: List of citations
        """
        return self.images.get(image_id, {}).get('citations', [])
    
    def generate_bibliography(self) -> str:
        """Generate a bibliography of all documents.
        
        Returns:
            str: Formatted bibliography
        """
        bibliography = []
        for doc_id, doc_data in self.documents.items():
            citations = doc_data.get('citations', [])
            if citations:
                bibliography.append(f"Document: {doc_id}")
                for citation in citations:
                    bibliography.append(f"- {citation['text']}")
                bibliography.append("")
        return "\n".join(bibliography)
    
    def generate_image_references(self) -> str:
        """Generate a list of image references.
        
        Returns:
            str: Formatted image references
        """
        references = []
        for image_id, image_data in self.images.items():
            citations = image_data.get('citations', [])
            if citations:
                references.append(f"Image: {image_id}")
                for citation in citations:
                    references.append(f"- {citation['text']}")
                references.append("")
        return "\n".join(references)
    
    def format_citation(self, doc_id: str, context: str) -> str:
        """
        Format a citation for a document.
        
        Args:
            doc_id: Document identifier
            context: Context of the citation
            
        Returns:
            Formatted citation
        """
        if doc_id not in self.documents:
            return f"[Unknown source: {doc_id}]"
            
        doc = self.documents[doc_id]
        return f"[{doc['path']}, {context}]"
    
    def format_image_citation(self, image_id: str, context: str) -> str:
        """
        Format a citation for an image.
        
        Args:
            image_id: Image identifier
            context: Context of the citation
            
        Returns:
            Formatted citation
        """
        if image_id not in self.images:
            return f"[Unknown image: {image_id}]"
            
        img = self.images[image_id]
        return f"[Figure {image_id}: {img['data']['alt_text']}, {context}]"

    def get_document_citation(self, doc_id: str) -> Optional[DocumentCitation]:
        """Get citations for a document.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            Optional[DocumentCitation]: Document citations if found
        """
        return self.doc_citations.get(doc_id)
    
    def get_image_citation(self, image_id: str) -> Optional[ImageCitation]:
        """Get citations for an image.
        
        Args:
            image_id (str): Image ID
            
        Returns:
            Optional[ImageCitation]: Image citations if found
        """
        return self.img_citations.get(image_id)
    
    def generate_bibliography_for_documents(self, doc_ids: List[str]) -> str:
        """Generate a bibliography for the given documents.
        
        Args:
            doc_ids (List[str]): List of document IDs
            
        Returns:
            str: Formatted bibliography
        """
        bibliography = []
        
        for doc_id in doc_ids:
            if doc_id in self.doc_citations:
                citation = self.doc_citations[doc_id]
                bibliography.append(f"[{doc_id}] {citation.title}")
                bibliography.append(f"    Path: {citation.path}")
                bibliography.append(f"    Sections: {', '.join(citation.sections)}")
                bibliography.append("")
                
        return "\n".join(bibliography)
    
    def generate_image_bibliography(self, img_ids: List[str]) -> str:
        """Generate a bibliography for the given images.
        
        Args:
            img_ids (List[str]): List of image IDs
            
        Returns:
            str: Formatted image bibliography
        """
        bibliography = []
        
        for img_id in img_ids:
            if img_id in self.img_citations:
                citation = self.img_citations[img_id]
                bibliography.append(f"[{img_id}] {citation.alt_text}")
                bibliography.append(f"    Path: {citation.path}")
                if citation.doc_id:
                    bibliography.append(f"    Document: {citation.doc_id}")
                bibliography.append("")
                
        return "\n".join(bibliography)
    
    def find_image(self, image_path: str) -> Optional[Dict]:
        """Find an image in the database based on its path or URL.
        
        Args:
            image_path (str): Path or URL of the image
            
        Returns:
            Optional[Dict]: Image information if found
        """
        self._log(f"\nDEBUG: Looking for image with path/URL: {image_path}")
        self._log(f"DEBUG: Current images in database: {list(self.images.keys())}")
        
        # First check if the image path matches any image in the database
        for img_id, img_info in self.images.items():
            self._log(f"DEBUG: Checking image {img_id}:")
            self._log(f"DEBUG: - Path: {img_info.get('path', 'N/A')}")
            
            # Check if the path matches
            if image_path == img_info.get('path'):
                self._log(f"DEBUG: Found matching image by path: {img_info}")
                return img_info
                
        # If not found by exact path, try to find by partial match
        for img_id, img_info in self.images.items():
            # Check if the image path contains the search term or vice versa
            if (image_path in img_info.get('path', '') or 
                img_info.get('path', '') in image_path):
                self._log(f"DEBUG: Found matching image by partial path: {img_info}")
                return img_info
                
        # If still not found, try to find by alt text
        for img_id, img_info in self.images.items():
            if image_path.lower() in img_info.get('alt_text', '').lower():
                self._log(f"DEBUG: Found matching image by alt text: {img_info}")
                return img_info
                
        self._log("DEBUG: No matching image found")
        return None 