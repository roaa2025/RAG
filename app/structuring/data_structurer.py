from typing import Dict, List, Any, Optional
import logging
import re
import json
from langchain.schema.document import Document

class DataStructurer:
    """
    Class for structuring preprocessed document text into organized formats.
    Handles layout recognition, content grouping, and structured data generation.
    """
    
    def __init__(self, structure_config: Optional[Dict] = None):
        """
        Initialize the data structurer with optional configuration.
        
        Args:
            structure_config: Optional configuration for data structuring
        """
        self.logger = logging.getLogger(__name__)
        self.structure_config = structure_config or {}
        self.logger.info("Initialized DataStructurer with config: %s", self.structure_config)
    
    def identify_layout_type(self, document: Document) -> str:
        """
        Identify the layout type of the document based on its content and metadata.
        
        Args:
            document: LangChain Document object
            
        Returns:
            Layout type identification (e.g., 'single-column', 'multi-column', 'table')
        """
        content = document.page_content
        metadata = document.metadata
        
        # Check if document has Docling metadata
        if metadata.get("dl_meta"):
            # Extract layout info from Docling metadata
            dl_meta = metadata.get("dl_meta", {})
            if isinstance(dl_meta, str):
                try:
                    dl_meta = json.loads(dl_meta)
                except json.JSONDecodeError:
                    dl_meta = {}
            
            # Check for table structures
            if "tables" in dl_meta or any("table" in str(item).lower() for item in dl_meta.get("doc_items", [])):
                return "table"
                
            # Check for multiple columns
            if any("column" in str(item).lower() for item in dl_meta.get("doc_items", [])):
                return "multi-column"
        
        # Fallback to content analysis
        # Check for table-like structure (multiple lines with similar delimiter patterns)
        lines = content.split("\n")
        if len(lines) > 3:
            delimiter_counts = [len(re.findall(r'\t|\s{3,}|,|\|', line)) for line in lines[:5]]
            if all(count > 0 for count in delimiter_counts) and max(delimiter_counts) - min(delimiter_counts) <= 1:
                return "table"
        
        # Default to single-column if no specific pattern detected
        return "single-column"
    
    def extract_tables(self, document: Document) -> List[Dict[str, Any]]:
        """
        Extract tabular data from the document if present.
        
        Args:
            document: LangChain Document object
            
        Returns:
            List of extracted tables as dictionaries
        """
        tables = []
        content = document.page_content
        metadata = document.metadata
        
        # Check if document has been identified as containing tables
        if self.identify_layout_type(document) != "table":
            return tables
            
        # Try to extract table from Docling metadata if available
        dl_meta = metadata.get("dl_meta", {})
        if isinstance(dl_meta, str):
            try:
                dl_meta = json.loads(dl_meta)
            except json.JSONDecodeError:
                dl_meta = {}
                
        # If Docling extracted tables, use them
        if "tables" in dl_meta:
            return dl_meta.get("tables", [])
            
        # Fallback to simple table parsing
        lines = content.split("\n")
        if len(lines) < 2:
            return tables
            
        # Try to detect delimiter
        potential_delimiters = ["\t", "|", ",", ";"]
        delimiter = max(potential_delimiters, key=lambda d: content.count(d)) if any(d in content for d in potential_delimiters) else None
        
        if delimiter:
            # Extract header and rows
            header = [col.strip() for col in lines[0].split(delimiter)]
            rows = []
            
            for line in lines[1:]:
                if line.strip():
                    cols = [col.strip() for col in line.split(delimiter)]
                    if len(cols) == len(header):
                        row_data = dict(zip(header, cols))
                        rows.append(row_data)
            
            if header and rows:
                tables.append({
                    "header": header,
                    "rows": rows
                })
                
        return tables
    
    def handle_multi_column(self, document: Document) -> Dict[str, Any]:
        """
        Process multi-column layouts and organize content accordingly.
        
        Args:
            document: LangChain Document object
            
        Returns:
            Structured data with columns properly identified
        """
        result = {"type": "multi-column", "content": []}
        content = document.page_content
        
        # Simple column detection based on content patterns
        # This is a simplified approach; for real-world use, consider more advanced techniques
        
        # Split content into lines
        lines = content.split("\n")
        
        # Initialize columns
        columns = []
        current_column = []
        
        for line in lines:
            # If empty line and we have content, it might indicate a column break
            if not line.strip() and current_column:
                columns.append("\n".join(current_column))
                current_column = []
            elif line.strip():
                current_column.append(line)
        
        # Add the last column if we have content
        if current_column:
            columns.append("\n".join(current_column))
        
        # If we couldn't detect columns, use the original content
        if not columns:
            result["content"] = [content]
        else:
            result["content"] = columns
            
        return result
    
    def structure_document(self, document: Document) -> Dict[str, Any]:
        """
        Convert a document into a structured format based on its layout.
        
        Args:
            document: LangChain Document object
            
        Returns:
            Structured representation of the document
        """
        layout_type = self.identify_layout_type(document)
        self.logger.info(f"Identified layout type: {layout_type} for document")
        
        # Create base structure with metadata
        result = {
            "content_type": layout_type,
            "metadata": document.metadata.copy(),
            "raw_text": document.page_content,
        }
        
        # Process based on layout type
        if layout_type == "table":
            tables = self.extract_tables(document)
            result["tables"] = tables
            result["structured_content"] = {"tables": tables}
            
        elif layout_type == "multi-column":
            column_data = self.handle_multi_column(document)
            result["columns"] = column_data["content"]
            result["structured_content"] = {"columns": column_data["content"]}
            
        else:  # single-column or other types
            # For single column, we keep the preprocessed text as is
            result["structured_content"] = {"text": document.page_content}
            
        return result
    
    def structure_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Structure the processed documents based on their content type and layout.
        
        Args:
            documents: List of processed Document objects
            
        Returns:
            List of structured document dictionaries
        """
        structured_documents = []
        
        for doc in documents:
            try:
                # Check if document is an image
                is_image = doc.metadata.get("is_image", False)
                
                if is_image:
                    # Handle image document
                    structured_doc = {
                        "content_type": "image",
                        "metadata": doc.metadata,
                        "raw_text": doc.page_content,
                        "image_info": {
                            "format": doc.metadata.get("file_type", ""),
                            "size": doc.metadata.get("image_processing", {}).get("size", 0),
                            "status": doc.metadata.get("image_processing", {}).get("status", "unknown")
                        }
                    }
                else:
                    # Handle regular document
                    structured_doc = {
                        "content_type": "text",
                        "metadata": doc.metadata,
                        "raw_text": doc.page_content,
                        "tables": self.extract_tables(doc),
                        "structured_content": self.handle_multi_column(doc)
                    }
                
                structured_documents.append(structured_doc)
                
            except Exception as e:
                self.logger.error(f"Error structuring document: {str(e)}")
                # If structuring fails, create a basic structure
                structured_documents.append({
                    "content_type": "unknown",
                    "metadata": doc.metadata,
                    "raw_text": doc.page_content
                })
                
        return structured_documents

class AdvancedDataStructurer(DataStructurer):
    """
    Advanced data structurer that adds sophisticated chunking capabilities.
    Inherits from DataStructurer and adds semantic and hierarchical chunking.
    """
    
    def __init__(self, structure_config: Optional[Dict] = None):
        """
        Initialize the advanced data structurer with chunking configuration.
        
        Args:
            structure_config: Configuration for data structuring and chunking
        """
        super().__init__(structure_config)
        self.chunking_config = self.structure_config.get("chunking", {})
        self.logger.info("Initialized AdvancedDataStructurer with chunking config: %s", self.chunking_config)
    
    def apply_fixed_size_chunking(self, text: str) -> List[Dict[str, Any]]:
        """
        Apply fixed-size chunking to the text.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunks with metadata
        """
        config = self.chunking_config.get("fixed_size", {})
        max_tokens = config.get("max_tokens", 1500)
        overlap_tokens = config.get("overlap_tokens", 150)
        
        # Simple implementation - split by words and create overlapping chunks
        words = text.split()
        chunks = []
        current_pos = 0
        
        while current_pos < len(words):
            chunk_end = min(current_pos + max_tokens, len(words))
            chunk = " ".join(words[current_pos:chunk_end])
            
            chunks.append({
                "content": chunk,
                "chunk_type": "fixed_size",
                "start_pos": current_pos,
                "end_pos": chunk_end,
                "token_count": chunk_end - current_pos
            })
            
            current_pos += (max_tokens - overlap_tokens)
            
        return chunks
    
    def apply_semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """
        Apply semantic chunking based on content similarity.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of semantically meaningful chunks
        """
        config = self.chunking_config.get("semantic", {})
        min_similarity = config.get("min_similarity", 0.7)
        max_chunk_size = config.get("max_chunk_size", 2000)
        
        # Simple implementation - split by paragraphs and group similar ones
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            if current_size + len(para.split()) > max_chunk_size and current_chunk:
                chunks.append({
                    "content": "\n\n".join(current_chunk),
                    "chunk_type": "semantic",
                    "size": current_size,
                    "paragraph_count": len(current_chunk)
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += len(para.split())
        
        if current_chunk:
            chunks.append({
                "content": "\n\n".join(current_chunk),
                "chunk_type": "semantic",
                "size": current_size,
                "paragraph_count": len(current_chunk)
            })
            
        return chunks
    
    def apply_hierarchical_chunking(self, text: str) -> List[Dict[str, Any]]:
        """
        Apply hierarchical chunking based on document structure.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of hierarchically structured chunks
        """
        config = self.chunking_config.get("hierarchical", {})
        max_depth = config.get("max_depth", 3)
        min_section_size = config.get("min_section_size", 500)
        
        # Simple implementation - split by headers and create hierarchy
        lines = text.split("\n")
        chunks = []
        current_section = []
        current_depth = 0
        
        for line in lines:
            if line.startswith("#"):
                if current_section:
                    chunks.append({
                        "content": "\n".join(current_section),
                        "chunk_type": "hierarchical",
                        "depth": current_depth,
                        "size": len(" ".join(current_section).split())
                    })
                current_section = []
                current_depth = min(line.count("#"), max_depth)
            else:
                current_section.append(line)
        
        if current_section:
            chunks.append({
                "content": "\n".join(current_section),
                "chunk_type": "hierarchical",
                "depth": current_depth,
                "size": len(" ".join(current_section).split())
            })
            
        return chunks
    
    def structure_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Structure documents with advanced chunking capabilities.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of structured documents with chunks
        """
        structured_documents = []
        
        for doc in documents:
            try:
                # Get base structure from parent class
                base_structure = super().structure_documents([doc])[0]
                
                # Apply different chunking strategies
                text = doc.page_content
                chunks = []
                
                # Fixed-size chunking
                fixed_chunks = self.apply_fixed_size_chunking(text)
                chunks.extend(fixed_chunks)
                
                # Semantic chunking
                semantic_chunks = self.apply_semantic_chunking(text)
                chunks.extend(semantic_chunks)
                
                # Hierarchical chunking
                hierarchical_chunks = self.apply_hierarchical_chunking(text)
                chunks.extend(hierarchical_chunks)
                
                # Add chunks to the structure
                base_structure["chunks"] = chunks
                structured_documents.append(base_structure)
                
            except Exception as e:
                self.logger.error(f"Error in advanced structuring: {str(e)}")
                # Fallback to basic structure if advanced structuring fails
                structured_documents.append(super().structure_documents([doc])[0])
        
        return structured_documents 