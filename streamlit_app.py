# Import configuration module first
import streamlit_config

# Standard library imports
import os
import sys
from pathlib import Path
from typing import Optional, List

# Import Streamlit
import streamlit as st

# Import application components
from markdown_processor import MarkdownRAG
from image_processor import ImageProcessor
from citation_handler import CitationHandler


def initialize_session_state():
    """Initialize session state variables."""
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'working_dir' not in st.session_state:
        st.session_state.working_dir = None
    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = None
    if 'citation_handler' not in st.session_state:
        st.session_state.citation_handler = None

def setup_components(working_dir: Path) -> None:
    """Set up RAG components.
    
    Args:
        working_dir (Path): Working directory path
    """
    # Create working directory if it doesn't exist
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    st.session_state.image_processor = ImageProcessor(
        max_size=(800, 800),
        quality=85,
        format="JPEG"
    )
    st.session_state.citation_handler = CitationHandler(
        working_dir=working_dir,
        verbose=True
    )
    st.session_state.rag = MarkdownRAG(
        working_dir=working_dir,
        image_processor=st.session_state.image_processor,
        citation_handler=st.session_state.citation_handler
    )
    st.session_state.working_dir = working_dir

def process_markdown_content(content: str, filename: str) -> None:
    """Process markdown content.
    
    Args:
        content (str): Markdown content
        filename (str): Original filename
    """
    if st.session_state.rag is None:
        st.error("Please initialize the system first")
        return
        
    try:
        # Save content to temporary file
        temp_file = Path(st.session_state.working_dir) / filename
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Process file
        st.session_state.rag.process_markdown_file(temp_file)
        st.success(f"Successfully processed {filename}")
        
        # Clean up
        temp_file.unlink()
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

def process_markdown_file(uploaded_file) -> None:
    """Process uploaded markdown file.
    
    Args:
        uploaded_file: Streamlit uploaded file
    """
    if st.session_state.rag is None:
        st.error("Please initialize the system first")
        return
        
    try:
        # Save uploaded file
        temp_file = Path(st.session_state.working_dir) / uploaded_file.name
        with open(temp_file, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            
        # Process file
        st.session_state.rag.process_markdown_file(temp_file)
        st.success(f"Successfully processed {uploaded_file.name}")
        
        # Clean up
        temp_file.unlink()
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

def display_response(response) -> None:
    """Display RAG response.
    
    Args:
        response: RAG response object
    """
    # Display text response
    st.markdown("### Response")
    st.write(response.text)
    
    # Display citations if available
    if response.citations:
        st.markdown("### Citations")
        st.write(response.citations)
        
    # Display images if available
    if response.images:
        st.markdown("### Images")
        for img_id in response.images:
            img_path = Path(st.session_state.working_dir) / 'images' / f"{img_id}.png"
            if img_path.exists():
                st.image(str(img_path))

def main():
    """Main Streamlit application."""
    st.title("LightRAG Markdown Agent")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for initialization
    with st.sidebar:
        st.header("Initialization")
        working_dir = st.text_input("Working Directory", value="./rag_data")
        
        if st.button("Initialize System"):
            setup_components(Path(working_dir))
            st.success("System initialized successfully!")
            
    # Main content area
    if st.session_state.rag is None:
        st.warning("Please initialize the system using the sidebar")
        return
        
    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Process Documents", "Query System"])
    
    # Process Documents tab
    with tab1:
        st.header("Process Documents")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Markdown File", type=['md', 'markdown'])
        if uploaded_file:
            process_markdown_file(uploaded_file)
            
        # Text input
        st.markdown("### Or Paste Markdown Content")
        content = st.text_area("Markdown Content", height=300)
        filename = st.text_input("Filename", value="content.md")
        
        if st.button("Process Content"):
            process_markdown_content(content, filename)
            
    # Query System tab
    with tab2:
        st.header("Query System")
        
        # Query input
        query = st.text_input("Enter your query")
        mode = st.selectbox(
            "Query Mode",
            options=['mix', 'local', 'global', 'hybrid', 'naive'],
            index=0
        )
        top_k = st.slider("Top K", min_value=1, max_value=100, value=60)
        
        if st.button("Submit Query"):
            if query:
                try:
                    response = st.session_state.rag.query(
                        query=query,
                        mode=mode,
                        top_k=top_k
                    )
                    display_response(response)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a query")

if __name__ == '__main__':
    main() 