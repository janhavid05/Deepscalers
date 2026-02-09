"""
File handling utilities for voice recognition

This module contains specialized functions for handling file operations
that are reliable across different operating systems, especially Windows.
"""

import os
import uuid
import logging
import tempfile
import shutil
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def get_media_root():
    """Get the absolute path to the media root directory"""
    # Go up one level from this file to get to the backend directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    media_root = os.path.join(backend_dir, 'media')
    
    # Ensure the directory exists
    os.makedirs(media_root, exist_ok=True)
    
    return media_root

def get_temp_audio_dir():
    """Get the absolute path to the temporary audio directory"""
    media_root = get_media_root()
    temp_dir = os.path.join(media_root, 'temp_audio')
    
    # Ensure the directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    return temp_dir

def normalize_path(path):
    """Convert path to absolute path with proper Windows separators"""
    # Convert to absolute path
    abs_path = os.path.abspath(path)
    
    # Convert to proper format for the current OS
    normalized = os.path.normpath(abs_path)
    
    return normalized

def save_uploaded_file(uploaded_file, prefix='audio_'):
    """
    Save an uploaded file to disk in a way that works reliably on Windows
    
    Args:
        uploaded_file: The UploadedFile object from request.FILES
        prefix: A prefix to add to the filename for organization
        
    Returns:
        The absolute path to the saved file
    """
    # Get the proper file extension
    original_name = uploaded_file.name
    _, file_ext = os.path.splitext(original_name)
    if not file_ext:
        # Default to .wav if no extension provided
        file_ext = '.wav'
    
    # Create a unique filename
    unique_id = str(uuid.uuid4())[:12]
    safe_filename = f"{prefix}{unique_id}{file_ext}"
    
    # Get a temporary directory that's guaranteed to be writable
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, safe_filename)
    
    # Log the path
    logger.info(f"Saving uploaded file to temporary location: {temp_file_path}")
    
    # Save the file using direct file operations
    try:
        with open(temp_file_path, 'wb') as destination:
            if uploaded_file.multiple_chunks():
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            else:
                destination.write(uploaded_file.read())
        
        # Verify file exists and has content
        if os.path.exists(temp_file_path):
            size = os.path.getsize(temp_file_path)
            logger.info(f"File saved successfully: {temp_file_path} ({size} bytes)")
            if size == 0:
                logger.error("File is empty (0 bytes)")
                os.unlink(temp_file_path)
                return None
        else:
            logger.error(f"File was not created at {temp_file_path}")
            return None
            
        return temp_file_path
        
    except Exception as e:
        logger.exception(f"Error saving file: {str(e)}")
        # Clean up if needed
        if os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return None

def cleanup_temp_file(file_path):
    """Safely remove a temporary file"""
    if not file_path or not os.path.exists(file_path):
        return
        
    try:
        os.unlink(file_path)
        logger.info(f"Deleted temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to delete temporary file {file_path}: {str(e)}") 