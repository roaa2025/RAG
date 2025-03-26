from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain.schema.document import Document
import os
from app.utils.document_utils import convert_webp_to_jpg

class DocumentOCR:
    """
    Class for handling OCR (Optical Character Recognition) extraction from various document types.
    Uses Docling directly for extracting text from different document formats.
    """

    def __init__(self, ocr_config: Optional[Dict] = None):
        """
        Initialize the OCR extractor with optional configuration.
        
        Args:
            ocr_config: Optional configuration for the OCR extraction process
        """
        self.logger = logging.getLogger(__name__)
        self.ocr_config = ocr_config or {}
        
        # Initialize DocumentConverter
        self.converter = DocumentConverter()
        self.logger.info("Initialized DocumentOCR with config: %s", self.ocr_config)

    def extract_text(self, file_path: Union[str, List[str]]) -> List[Document]:
        """
        Extract text from documents using Docling's OCR capabilities.
        
        Args:
            file_path: Path to the document or list of document paths
            
        Returns:
            List of LangChain Document objects containing the extracted text
        """
        # Convert single file path to list for consistent processing
        if isinstance(file_path, str):
            file_path = [file_path]
            
        self.logger.info(f"Found {len(file_path)} files to process")
        
        try:
            documents = []
            errors = []
            processed_count = 0
            converted_files = []
            
            for path in file_path:
                try:
                    # Check if file exists
                    if not os.path.exists(path):
                        self.logger.error(f"File not found: {path}")
                        continue
                        
                    # Check file extension
                    file_ext = Path(path).suffix.lower()
                    
                    # Handle text files directly
                    if file_ext == '.txt':
                        with open(path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        # Create LangChain Document object with metadata
                        metadata = {
                            "source": path,
                            "file_type": file_ext,
                            "dl_meta": {"text_file": True},
                            "is_image": False
                        }
                        
                        document = Document(
                            page_content=text,
                            metadata=metadata
                        )
                        documents.append(document)
                        processed_count += 1
                        continue
                    
                    # Convert webp files to jpg
                    if file_ext == '.webp':
                        self.logger.info(f"Converting WebP file to JPG: {path}")
                        try:
                            converted_path = convert_webp_to_jpg(path)
                            converted_files.append((path, converted_path))
                            path = converted_path
                            file_ext = '.jpg'
                        except Exception as conv_error:
                            self.logger.error(f"Error converting WebP file {path}: {str(conv_error)}")
                            errors.append(f"{path}: Conversion failed - {str(conv_error)}")
                            continue
                    
                    # Log processing attempt
                    self.logger.info(f"Processing file: {path} (type: {file_ext})")
                    
                    # Process each file individually
                    result = self.converter.convert(path)
                    
                    # Extract text from the document
                    text = result.document.export_to_markdown()
                    
                    # Create LangChain Document object with enhanced metadata
                    metadata = {
                        "source": path,
                        "file_type": file_ext,
                        "dl_meta": result.document.model_dump(),
                        "is_image": file_ext in ['.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff']
                    }
                    
                    # Add image-specific metadata if it's an image
                    if metadata["is_image"]:
                        metadata["image_processing"] = {
                            "status": "processed",
                            "format": file_ext,
                            "size": os.path.getsize(path),
                            "original_format": "webp" if file_ext == '.jpg' and path in [c[1] for c in converted_files] else file_ext
                        }
                        self.logger.info(f"Successfully processed image: {path}")
                    
                    document = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    documents.append(document)
                    processed_count += 1
                    
                except Exception as doc_error:
                    self.logger.error(f"Error processing document {path}: {str(doc_error)}")
                    errors.append(f"{path}: {str(doc_error)}")
                    continue
            
            # Clean up converted files
            for original_path, converted_path in converted_files:
                try:
                    os.remove(converted_path)
                    self.logger.info(f"Cleaned up converted file: {converted_path}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up converted file {converted_path}: {str(cleanup_error)}")
            
            if not documents and errors:
                raise ValueError(f"Failed to process any documents. Errors: {', '.join(errors)}")
                
            self.logger.info(f"Successfully processed {processed_count} out of {len(file_path)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            raise 