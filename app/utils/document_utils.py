from typing import Dict, List, Any, Optional, Union
import os
import logging
import json
import tempfile
from pathlib import Path
import mimetypes
from PIL import Image

logger = logging.getLogger(__name__)

def get_file_mimetype(file_path: str) -> str:
    """
    Determine the MIME type of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if not mime_type:
        # Check file extension for common types
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tif': 'image/tiff',
            '.tiff': 'image/tiff',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html'
        }
        mime_type = mime_map.get(ext, 'application/octet-stream')
    
    return mime_type

def is_image_file(file_path: str) -> bool:
    """
    Check if a file is an image based on its MIME type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is an image, False otherwise
    """
    mime_type = get_file_mimetype(file_path)
    return mime_type and mime_type.startswith('image/')

def is_pdf_file(file_path: str) -> bool:
    """
    Check if a file is a PDF based on its MIME type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a PDF, False otherwise
    """
    mime_type = get_file_mimetype(file_path)
    return mime_type == 'application/pdf'

def is_office_document(file_path: str) -> bool:
    """
    Check if a file is a Microsoft Office or OpenOffice document.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is an office document, False otherwise
    """
    mime_type = get_file_mimetype(file_path)
    office_types = [
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-powerpoint',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.oasis.opendocument.text',
        'application/vnd.oasis.opendocument.spreadsheet',
        'application/vnd.oasis.opendocument.presentation'
    ]
    return mime_type in office_types

def save_to_json(data: Any, output_path: str) -> str:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        output_path: Path to save the JSON file
        
    Returns:
        Path to the saved file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Data successfully saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving data to JSON: {str(e)}")
        raise

def create_output_filename(input_path: str, output_dir: Optional[str] = None, suffix: str = '_processed', extension: str = '.json') -> str:
    """
    Create an output filename based on the input filename.
    
    Args:
        input_path: Path to the input file
        output_dir: Directory to save the output file (defaults to same as input)
        suffix: Suffix to add to the filename
        extension: File extension for the output file
        
    Returns:
        Output file path
    """
    input_path = os.path.abspath(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}{suffix}{extension}")
    else:
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f"{base_name}{suffix}{extension}")
    
    return output_path

def get_temp_file_path(suffix: str = '.tmp') -> str:
    """
    Create a temporary file and return its path.
    
    Args:
        suffix: File extension for the temporary file
        
    Returns:
        Path to the temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()
    return temp_file.name

def ensure_valid_input(file_path: Union[str, List[str]]) -> List[str]:
    """
    Validate input file paths and return a list of valid paths.
    
    Args:
        file_path: Single file path or list of file paths
        
    Returns:
        List of valid file paths
    """
    if isinstance(file_path, str):
        file_paths = [file_path]
    else:
        file_paths = file_path
    
    valid_paths = []
    
    for path in file_paths:
        if os.path.isfile(path) or path.startswith(('http://', 'https://')):
            valid_paths.append(path)
        else:
            logger.warning(f"Invalid file path: {path}")
    
    if not valid_paths:
        raise ValueError("No valid file paths provided")
    
    return valid_paths

def convert_webp_to_jpg(file_path: str, output_dir: str = None) -> str:
    """
    Convert a WebP image to JPG format.
    
    Args:
        file_path: Path to the WebP file
        output_dir: Optional directory to save the converted file. If None, uses the same directory as input.
        
    Returns:
        Path to the converted JPG file
    """
    try:
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, Path(file_path).stem + '.jpg')
        else:
            output_path = str(Path(file_path).with_suffix('.jpg'))
            
        # Open and convert the image
        with Image.open(file_path) as img:
            # Convert to RGB if necessary (for RGBA images)
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save as JPG
            img.save(output_path, 'JPEG', quality=95)
            logger.info(f"Successfully converted {file_path} to {output_path}")
            return output_path
            
    except Exception as e:
        logger.error(f"Error converting {file_path} to JPG: {str(e)}")
        raise

def save_structured_data(data: List[Dict[str, Any]], output_path: str) -> None:
    """Save structured data to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {str(e)}")
        raise 