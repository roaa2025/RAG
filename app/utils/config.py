from typing import Dict, Any
import logging
import os
import json

logger = logging.getLogger(__name__)

# Default OCR configuration
DEFAULT_OCR_CONFIG = {
    # Docling OCR settings
    "ocr": True,
    "ocr_lang": "eng",  # Default to English
    "ocr_dpi": 300,     # Default DPI for OCR
    "force_ocr": True,  # Force OCR on all documents to ensure image processing
    "image_processing": {
        "enabled": True,
        "supported_formats": [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"],
        "max_image_size": 10000000,  # 10MB
        "preserve_original": True
    }
}

# Default text preprocessing configuration
DEFAULT_PREPROCESSING_CONFIG = {
    # OCR error corrections mapping
    "ocr_corrections": {
        "l1": "h",
        "rn": "m",
        "0": "O",
        "1": "I",
        # Add more common OCR errors as needed
    },
    
    # Language detection parameters
    "language_detection": {
        "min_text_length": 10,
        "default_language": "en"
    },
    
    # Normalization settings
    "normalization": {
        "lowercase": True,
        "remove_extra_whitespace": True,
        "remove_special_chars": False
    }
}

# Default data structuring configuration
DEFAULT_STRUCTURING_CONFIG = {
    # Table extraction settings
    "table_extraction": {
        "max_header_rows": 2,
        "detect_delimiters": True,
        "potential_delimiters": ["\t", "|", ",", ";"]
    },
    
    # Layout settings
    "layout": {
        "detect_columns": True,
        "min_column_width": 50  # Minimum width in pixels to consider as a column
    }
}

# Default skill extraction configuration
DEFAULT_SKILL_EXTRACTION_CONFIG = {
    "enabled": True,
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "method": "openai",  # Options: 'openai', 'spacy'
    "model": "gpt-4o",  # Use gpt-4o model
    "min_confidence": 0.5,
    "store_rejected": True,
    "custom_skills_list": []  # Optional list of domain-specific skills to recognize
}

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the document parser.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        "ocr": {
            "convert_kwargs": {
                "use_ocr": True,
                "ocr_lang": "eng",
                "ocr_dpi": 300,
                "force_ocr": False
            }
        },
        "preprocessing": {
            "ocr_corrections": {
                "l1": "h",
                "rn": "m"
            },
            "normalization": {
                "lowercase": True,
                "remove_extra_whitespace": True
            }
        },
        "structuring": {
            "use_advanced_chunking": False,  # Set to True to use multi-layer chunking
            "table_extraction": {
                "detect_delimiters": True
            },
            "layout": {
                "detect_columns": True
            },
            "chunking": {
                "hierarchical": {
                    "enabled": True,
                    "min_section_length": 100,
                    "headings_regexp": r'^#+\s+.+|^.+\n[=\-]+$'
                },
                "semantic": {
                    "enabled": True,
                    "similarity_threshold": 0.75,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "min_sentences_per_chunk": 3,
                    "max_sentences_to_consider": 20
                },
                "fixed_size": {
                    "enabled": True,
                    "max_tokens": 1000,
                    "max_chars": 4000,
                    "overlap_tokens": 100
                },
                "metadata": {
                    "preserve_headers": True,
                    "include_position": True,
                    "include_chunk_type": True
                }
            }
        },
        "skill_extraction": {
            "enabled": True,
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "method": "openai",
            "model": "gpt-4o",  # Use gpt-4o model
            "min_confidence": 0.5,
            "store_rejected": True
        }
    }

def merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge user config with default config.
    
    Args:
        default_config: Default configuration dictionary
        user_config: User-provided configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary loaded from file
    """
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            
        default_config = get_default_config()
        merged_config = merge_configs(default_config, user_config)
        
        logger.info(f"Loaded configuration from {config_path}")
        return merged_config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        # Return default config if loading fails
        return get_default_config()

def save_config_to_file(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        return False 