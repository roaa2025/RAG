from typing import Dict, List, Optional, Union
import re
import logging
from langdetect import detect, LangDetectException
from langchain.schema.document import Document

class TextPreprocessor:
    """
    Class for preprocessing extracted text from documents.
    Handles text cleaning, noise removal, OCR error correction, and language detection.
    """
    
    def __init__(self, preprocessing_config: Optional[Dict] = None):
        """
        Initialize the text preprocessor with optional configuration.
        
        Args:
            preprocessing_config: Optional configuration for text preprocessing
        """
        self.logger = logging.getLogger(__name__)
        self.preprocessing_config = preprocessing_config or {}
        self.logger.info("Initialized TextPreprocessor with config: %s", self.preprocessing_config)
    
    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing noise and correcting common OCR errors.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters
        cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text)
        
        # Correct common OCR errors (examples, can be expanded)
        ocr_corrections = {
            'l1': 'h', # Common OCR mistake
            'rn': 'm',  # Common OCR mistake
            '0': 'O',   # Common OCR mistake for certain fonts
        }
        
        # Apply corrections from config if provided
        if self.preprocessing_config.get("ocr_corrections"):
            ocr_corrections.update(self.preprocessing_config.get("ocr_corrections"))
            
        for error, correction in ocr_corrections.items():
            # Only correct standalone instances to avoid false positives
            cleaned_text = re.sub(r'\b' + error + r'\b', correction, cleaned_text)
            
        return cleaned_text
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text for language detection
            
        Returns:
            ISO language code or None if detection fails
        """
        if not text or len(text.strip()) < 10:
            self.logger.warning("Text too short for reliable language detection")
            return None
            
        try:
            language = detect(text)
            self.logger.debug(f"Detected language: {language}")
            return language
        except LangDetectException as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            return None
    
    def normalize_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Normalize text based on detected language and configuration.
        
        Args:
            text: Input text to normalize
            language: ISO language code (optional)
            
        Returns:
            Normalized text
        """
        # Detect language if not provided
        if not language:
            language = self.detect_language(text) or 'en'
        
        # Apply language-specific normalization
        if language == 'en':
            # English-specific normalization
            normalized_text = text.lower()
        else:
            # Default normalization
            normalized_text = text
            
        return normalized_text
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process a list of documents by cleaning and normalizing their content.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of processed Document objects
        """
        processed_documents = []
        
        for doc in documents:
            try:
                # Clean the text
                cleaned_text = self.clean_text(doc.page_content)
                
                # Detect language
                language = self.detect_language(cleaned_text)
                
                # Normalize text
                normalized_text = self.normalize_text(cleaned_text, language)
                
                # Create a new document with processed content
                # Add language metadata if detected
                metadata = doc.metadata.copy()
                if language:
                    metadata["language"] = language
                    
                metadata["preprocessing_applied"] = True
                
                processed_doc = Document(
                    page_content=normalized_text,
                    metadata=metadata
                )
                
                processed_documents.append(processed_doc)
                
            except Exception as e:
                self.logger.error(f"Error processing document: {str(e)}")
                # If processing fails, keep the original document
                processed_documents.append(doc)
                
        self.logger.info(f"Successfully processed {len(processed_documents)} documents")
        return processed_documents 