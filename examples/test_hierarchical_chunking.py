import os
import json
import logging
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Simple Document class for testing
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Custom sentence tokenizer
def custom_sentence_tokenize(text):
    """Simple sentence tokenizer using regex patterns"""
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    
    # Further split very long sentences
    result = []
    for sentence in sentences:
        if len(sentence) > 300:
            sub_parts = re.split(r'(?<=[;:])\s+', sentence)
            result.extend(sub_parts)
        else:
            result.append(sentence)
    
    return [s for s in result if s.strip()]

# Simplified version of MultiLayerChunker just for hierarchical chunking
class SimpleHierarchicalChunker:
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "min_section_length": 100,
            "headings_regexp": r'^#+\s+.+|^.+\n[=\-]+$'
        }
        
        # Merge config
        self.merged_config = {**self.default_config, **self.config}
        self.logger.info(f"Initialized chunker with config: {self.merged_config}")
    
    def _extract_headers_and_sections(self, document):
        """Extract headers and sections from document based on markdown structure"""
        sections = []
        content = document.page_content
        header_pattern = self.merged_config["headings_regexp"]
        min_section_length = self.merged_config["min_section_length"]
        
        # Look for headers using regex
        header_matches = list(re.finditer(header_pattern, content, re.MULTILINE))
        
        if header_matches:
            # Process each section (from one header to the next)
            for i, match in enumerate(header_matches):
                header = match.group(0).strip()
                header_start = match.start()
                
                # Get header level (number of # for markdown)
                level = 1
                if header.startswith('#'):
                    level = len(re.match(r'^#+', header).group(0))
                    header = re.sub(r'^#+\s+', '', header)
                    
                # Get section content (from end of current header to start of next, or end of document)
                if i < len(header_matches) - 1:
                    section_end = header_matches[i + 1].start()
                else:
                    section_end = len(content)
                    
                section_content = content[header_start + len(match.group(0)):section_end].strip()
                
                # Only include sections that have meaningful content
                if len(section_content) >= min_section_length:
                    sections.append({
                        "header": header,
                        "content": section_content,
                        "level": level,
                        "start_pos": header_start,
                        "end_pos": section_end
                    })
        
        # If no headers found, treat the whole document as one section
        if not sections:
            sections.append({
                "header": "Document",
                "content": content,
                "level": 0,
                "start_pos": 0,
                "end_pos": len(content)
            })
            
        return sections
    
    def chunk_document(self, document):
        """Apply hierarchical chunking to a document"""
        chunked_docs = []
        
        try:
            # Extract sections based on document structure
            sections = self._extract_headers_and_sections(document)
            
            for section in sections:
                # Prepare metadata
                section_metadata = document.metadata.copy()
                section_metadata["section_header"] = section["header"]
                section_metadata["section_level"] = section["level"]
                section_metadata["section_start_pos"] = section["start_pos"]
                section_metadata["section_end_pos"] = section["end_pos"]
                section_metadata["chunk_type"] = "hierarchical"
                
                # Create new Document for section
                chunked_docs.append(Document(
                    page_content=section["content"],
                    metadata=section_metadata
                ))
                
            return chunked_docs
            
        except Exception as e:
            self.logger.error(f"Error in document chunking: {str(e)}")
            # Return the original document as one chunk on error
            doc_metadata = document.metadata.copy()
            doc_metadata["chunk_type"] = "original"
            doc_metadata["chunking_error"] = str(e)
            return [Document(
                page_content=document.page_content,
                metadata=doc_metadata
            )]
    
    def _token_count(self, text):
        """Approximate token count (4 chars per token)"""
        return len(text) // 4
    
    def evaluate_chunks(self, chunks):
        """Evaluate the quality of the chunking"""
        if not chunks:
            return {"error": "No chunks to evaluate"}
            
        try:
            # Calculate metrics
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            token_counts = [self._token_count(chunk.page_content) for chunk in chunks]
            
            # Collect metadata statistics
            chunk_types = [chunk.metadata.get("chunk_type", "unknown") for chunk in chunks]
            type_counts = defaultdict(int)
            for chunk_type in chunk_types:
                type_counts[chunk_type] += 1
                
            # Build evaluation results
            evaluation = {
                "chunk_count": len(chunks),
                "chunk_sizes": {
                    "min": min(chunk_sizes),
                    "max": max(chunk_sizes),
                    "avg": sum(chunk_sizes) / len(chunk_sizes),
                    "std": np.std(chunk_sizes)
                },
                "token_counts": {
                    "min": min(token_counts),
                    "max": max(token_counts),
                    "avg": sum(token_counts) / len(token_counts),
                    "std": np.std(token_counts)
                },
                "chunk_types": dict(type_counts)
            }
                
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating chunks: {str(e)}")
            return {
                "error": f"Error evaluating chunks: {str(e)}",
                "chunk_count": len(chunks)
            }

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("output") / "hierarchical_chunking_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test document
    test_doc_path = Path("examples") / "documents" / "test_document.md"
    
    try:
        with open(test_doc_path, 'r', encoding='utf-8') as f:
            document_content = f.read()
        
        # Create document object
        doc = Document(
            page_content=document_content,
            metadata={"source": str(test_doc_path), "doc_id": "document_0"}
        )
        
        # Load configuration from file if exists, or use default config
        config_path = Path("examples") / "advanced_chunking_config.json"
        chunker_config = {}
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "structuring" in config and "chunking" in config["structuring"] and "hierarchical" in config["structuring"]["chunking"]:
                    chunker_config = config["structuring"]["chunking"]["hierarchical"]
        
        # Initialize chunker
        chunker = SimpleHierarchicalChunker(chunker_config)
        
        # Apply chunking
        chunks = chunker.chunk_document(doc)
        
        # Evaluate chunks
        evaluation = chunker.evaluate_chunks(chunks)
        
        # Prepare results
        all_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"document_0_chunk_{i}"
            chunk.metadata["chunk_id"] = chunk_id
            
            # Convert to dict for JSON serialization
            chunk_dict = {
                "chunk_id": chunk_id,
                "document_id": "document_0",
                "content": chunk.page_content,
                "content_type": "text",
                "chunk_type": chunk.metadata.get("chunk_type", "unknown"),
                "section_header": chunk.metadata.get("section_header", ""),
                "section_level": chunk.metadata.get("section_level", 0)
            }
            
            all_chunks.append(chunk_dict)
        
        # Save results
        results = {
            "document_count": 1,
            "total_chunk_count": len(all_chunks),
            "chunks": all_chunks,
            "chunk_evaluation": {"document_0": evaluation}
        }
        
        # Save to JSON file
        output_file = output_dir / "test_document_processed.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\nHierarchical Chunking complete!")
        print(f"Generated {len(all_chunks)} chunks from document")
        
        # Print token statistics
        if "token_counts" in evaluation:
            token_stats = evaluation["token_counts"]
            print(f"Token counts: min={token_stats.get('min', 0)}, "
                  f"max={token_stats.get('max', 0)}, "
                  f"avg={token_stats.get('avg', 0):.2f}")
        
        # Print headers of chunks
        print("\nChunks by section:")
        for i, chunk in enumerate(chunks):
            print(f"{i+1}. {chunk.metadata.get('section_header')} (Level {chunk.metadata.get('section_level', 0)})")
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main() 