"""
Streamlit configuration module.
This module should be imported before any other modules to ensure proper configuration.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_config")

# Disable Streamlit's file watcher completely to avoid PyTorch issues
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_FOLDER_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_WATCHDOG_DISABLE'] = 'true'

# Disable PyTorch JIT
os.environ['PYTORCH_JIT'] = '0'

# Try to disable torch module scanning
try:
    # Import the monkey patch for file watchdog
    import streamlit.watcher.local_sources_watcher as watcher
    
    # Save original function
    original_get_module_paths = getattr(watcher, 'get_module_paths', None)
    
    # Define safe function
    def safe_get_module_paths(module):
        """Safely get module paths, avoiding torch modules."""
        module_name = getattr(module, '__name__', '')
        
        # Skip torch modules
        if module_name.startswith('torch'):
            return []
            
        # Use original function for non-torch modules
        if original_get_module_paths:
            try:
                return original_get_module_paths(module)
            except Exception:
                return []
        
        # Fallback
        return []
    
    # Apply monkey patch if we found the function
    if original_get_module_paths:
        setattr(watcher, 'get_module_paths', safe_get_module_paths)
        logger.info("Successfully patched Streamlit watcher for torch compatibility")
except Exception as e:
    logger.warning(f"Could not patch Streamlit watcher: {e}")

logger.info("Streamlit configuration loaded") 