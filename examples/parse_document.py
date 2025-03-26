"""
Example script demonstrating how to use the document parser.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path so we can import the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import parse_document

def main():
    """
    Example of using the document parser with a custom configuration.
    """
    # Define a custom configuration
    custom_config = {
        "ocr": {
            "convert_kwargs": {
                "ocr_lang": "eng",  # English
                "ocr_dpi": 400,     # Higher quality OCR
                "force_ocr": False
            }
        },
        "preprocessing": {
            "normalization": {
                "lowercase": True,
                "remove_extra_whitespace": True
            }
        },
        "structuring": {
            "layout": {
                "detect_columns": True
            }
        }
    }
    
    # Check if a document path was provided
    if len(sys.argv) < 2:
        print("Please provide a document path as an argument")
        print("Usage: python examples/parse_document.py <document_path>")
        return
    
    # Get document path from command line arguments
    document_path = sys.argv[1]
    
    # Get optional output directory from command line arguments
    output_dir = None
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        
    print(f"Processing document: {document_path}")
    print(f"Output directory: {output_dir or 'None (results will not be saved)'}")
    print(f"Using custom configuration: {json.dumps(custom_config, indent=2)}")
    
    # Parse the document
    result = parse_document(document_path, config=custom_config, output_dir=output_dir)
    
    # Print a summary of the results
    print("\nDocument parsing complete!")
    print(f"Processed {result['document_count']} document(s) into {result['chunk_count']} chunks")
    
    if "output_path" in result:
        print(f"Results saved to: {result['output_path']}")
    else:
        # Print a truncated version of the first chunk to show it worked
        if result["processed_documents"]:
            first_doc = result["processed_documents"][0]
            content_type = first_doc.get("content_type", "unknown")
            raw_text = first_doc.get("raw_text", "")
            
            print(f"\nFirst document chunk (type: {content_type}):")
            print("-" * 50)
            print(raw_text[:500] + ("..." if len(raw_text) > 500 else ""))
            print("-" * 50)

if __name__ == "__main__":
    main() 