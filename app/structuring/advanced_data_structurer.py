from typing import Dict, List, Any, Optional
import logging
from langchain.schema import Document
from app.structuring.data_structurer import DataStructurer
from app.structuring.chunking_strategy import MultiLayerChunker

class AdvancedDataStructurer(DataStructurer):
    """
    Enhanced data structurer that incorporates the multi-layer chunking strategy
    for improved document processing. Extends the basic DataStructurer with advanced
    chunking capabilities.
    """
    
    def __init__(self, structure_config: Optional[Dict] = None):
        """
        Initialize the advanced data structurer with optional configuration.
        
        Args:
            structure_config: Optional configuration for data structuring and chunking
        """
        super().__init__(structure_config)
        self.logger = logging.getLogger(__name__)
        
        # Extract chunking configuration from structure_config
        chunking_config = self.structure_config.get("chunking", {})
        
        # Initialize the multi-layer chunker
        self.chunker = MultiLayerChunker(config=chunking_config)
        self.logger.info("Initialized AdvancedDataStructurer with multi-layer chunking support")
    
    def structure_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Apply the multi-layer chunking strategy to documents and structure them.
        
        Args:
            documents: List of preprocessed Document objects
            
        Returns:
            Dictionary with structured document data, including chunk metrics
        """
        chunked_documents = []
        all_chunks = []
        chunk_evaluations = {}
        chunk_metrics = {
            "total_count": 0,
            "by_type": {},
            "by_document": {},
            "average_chunk_size": 0,
            "chunk_size_distribution": {
                "small": 0,  # <500 chars
                "medium": 0, # 500-1500 chars
                "large": 0   # >1500 chars
            }
        }
        
        self.logger.info(f"Processing {len(documents)} documents with advanced chunking strategy")
        
        for i, doc in enumerate(documents):
            try:
                # Apply multi-layer chunking to the document
                self.logger.info(f"Applying multi-layer chunking to document {i+1}/{len(documents)}")
                document_chunks = self.chunker.chunk_document(doc)
                
                # Evaluate chunks for quality
                chunk_evaluation = self.chunker.evaluate_chunks(document_chunks)
                chunk_evaluations[f"document_{i}"] = chunk_evaluation
                
                # Store chunk IDs for metadata
                chunk_ids = [f"doc_{i}_chunk_{j}" for j in range(len(document_chunks))]
                
                # Track document-specific chunk metrics
                doc_id = doc.metadata.get("doc_id", f"doc_{i}")
                chunk_metrics["by_document"][doc_id] = {
                    "count": len(document_chunks)
                }
                
                # Update total chunk count
                chunk_metrics["total_count"] += len(document_chunks)
                
                # Analyze chunk types and sizes
                for chunk in document_chunks:
                    # Add to all chunks list
                    all_chunks.append(chunk)
                    
                    # Track chunk type
                    chunk_type = chunk.metadata.get("chunk_type", "unknown")
                    if chunk_type not in chunk_metrics["by_type"]:
                        chunk_metrics["by_type"][chunk_type] = 0
                    chunk_metrics["by_type"][chunk_type] += 1
                    
                    # Track chunk size distribution
                    content_length = len(chunk.page_content)
                    if content_length < 500:
                        chunk_metrics["chunk_size_distribution"]["small"] += 1
                    elif content_length < 1500:
                        chunk_metrics["chunk_size_distribution"]["medium"] += 1
                    else:
                        chunk_metrics["chunk_size_distribution"]["large"] += 1
                
                # Structure the original document (for overview)
                is_image = doc.metadata.get("is_image", False)
                
                if is_image:
                    # Handle image document
                    original_doc_structure = {
                        "content_type": "image",
                        "metadata": doc.metadata,
                        "raw_text": doc.page_content,
                        "image_info": {
                            "format": doc.metadata.get("file_type", ""),
                            "size": doc.metadata.get("image_processing", {}).get("size", 0),
                            "status": doc.metadata.get("image_processing", {}).get("status", "unknown")
                        },
                        "chunked": True,
                        "chunk_count": len(document_chunks),
                        "chunk_ids": chunk_ids,
                        "chunk_evaluation": chunk_evaluation
                    }
                else:
                    # Handle regular document
                    layout_type = self.identify_layout_type(doc)
                    original_doc_structure = {
                        "content_type": layout_type,
                        "metadata": doc.metadata,
                        "raw_text": doc.page_content,
                        "chunked": True,
                        "chunk_count": len(document_chunks),
                        "chunk_ids": chunk_ids,
                        "chunk_evaluation": chunk_evaluation
                    }
                    
                    # Add additional structure for tables if present
                    if layout_type == "table":
                        original_doc_structure["tables"] = self.extract_tables(doc)
                
                chunked_documents.append(original_doc_structure)
                
            except Exception as e:
                self.logger.error(f"Error chunking document {i}: {str(e)}")
                # If chunking fails, use the base structuring method
                structured_doc = super().structure_document(doc)
                structured_doc["chunked"] = False
                structured_doc["chunking_error"] = str(e)
                chunked_documents.append(structured_doc)
        
        # Calculate average chunk size
        if all_chunks:
            total_chars = sum(len(chunk.page_content) for chunk in all_chunks)
            chunk_metrics["average_chunk_size"] = total_chars / len(all_chunks)
        
        # Structure all individual chunks
        structured_chunks = []
        
        for i, chunk in enumerate(all_chunks):
            try:
                # Extract chunk-specific metadata
                chunk_type = chunk.metadata.get("chunk_type", "unknown")
                
                # Structure the chunk based on its layout type
                layout_type = self.identify_layout_type(chunk)
                
                structured_chunk = {
                    "chunk_id": f"chunk_{i}",
                    "content_type": layout_type,
                    "chunk_type": chunk_type,
                    "metadata": chunk.metadata,
                    "content": chunk.page_content
                }
                
                # Add section info if available
                if "section_header" in chunk.metadata:
                    structured_chunk["section_header"] = chunk.metadata["section_header"]
                    structured_chunk["section_level"] = chunk.metadata.get("section_level", 0)
                
                # Add positioning info
                if "semantic_chunk_index" in chunk.metadata:
                    structured_chunk["semantic_info"] = {
                        "index": chunk.metadata["semantic_chunk_index"],
                        "total": chunk.metadata["semantic_chunk_count"]
                    }
                
                if "fixed_chunk_index" in chunk.metadata:
                    structured_chunk["fixed_info"] = {
                        "index": chunk.metadata["fixed_chunk_index"],
                        "total": chunk.metadata["fixed_chunk_count"]
                    }
                
                # Extract tables if present
                if layout_type == "table":
                    structured_chunk["tables"] = self.extract_tables(chunk)
                
                structured_chunks.append(structured_chunk)
                
            except Exception as e:
                self.logger.error(f"Error structuring chunk {i}: {str(e)}")
                # If structuring fails, create a basic structure
                structured_chunks.append({
                    "chunk_id": f"chunk_{i}",
                    "content_type": "unknown",
                    "chunk_type": chunk.metadata.get("chunk_type", "unknown"),
                    "metadata": chunk.metadata,
                    "content": chunk.page_content,
                    "structuring_error": str(e)
                })
        
        # Add evaluation summary
        overall_evaluation = {
            "document_count": len(documents),
            "total_chunk_count": len(all_chunks),
            "chunk_metrics": chunk_metrics,
            "chunk_evaluation": chunk_evaluations,
            "documents": chunked_documents,
            "chunks": structured_chunks
        }
        
        # Return the structured documents and chunks
        return overall_evaluation 