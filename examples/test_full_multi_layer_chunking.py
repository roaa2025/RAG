import os
import json
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    # Import our chunking strategy
    from app.structuring.chunking_strategy import MultiLayerChunker
    
    # Import document-related classes
    from langchain.schema.document import Document
except ImportError as e:
    logging.error(f"Error importing required modules: {e}")
    logging.info("This script assumes you have the necessary packages installed (langchain, transformers, etc.)")
    sys.exit(1)

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("output") / "multi_layer_chunking_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration from file
    config_path = Path("examples") / "advanced_chunking_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            chunking_config = config.get("structuring", {}).get("chunking", {})
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        chunking_config = {}  # Use default configuration
    
    # Log configuration settings
    logging.info(f"Loaded chunking configuration")
    logging.info(f"Hierarchical chunking enabled: {chunking_config.get('hierarchical', {}).get('enabled', True)}")
    logging.info(f"Semantic chunking enabled: {chunking_config.get('semantic', {}).get('enabled', True)}")
    logging.info(f"Fixed-size chunking enabled: {chunking_config.get('fixed_size', {}).get('enabled', True)}")
    
    # Load test document
    test_doc_path = Path("examples") / "documents" / "test_document.md"
    
    try:
        with open(test_doc_path, 'r', encoding='utf-8') as f:
            document_content = f.read()
        
        logging.info(f"Loaded test document: {test_doc_path}")
        logging.info(f"Document size: {len(document_content)} characters")
        
        # Create document object
        doc = Document(
            page_content=document_content,
            metadata={"source": str(test_doc_path), "doc_id": "document_0"}
        )
    except Exception as e:
        logging.error(f"Error loading test document: {e}")
        sys.exit(1)
    
    # Initialize chunker
    try:
        chunker = MultiLayerChunker(chunking_config)
        logging.info("Initialized MultiLayerChunker")
    except Exception as e:
        logging.error(f"Error initializing chunker: {e}")
        sys.exit(1)
    
    # Apply chunking
    try:
        logging.info("Applying multi-layer chunking strategy...")
        chunks = chunker.chunk_document(doc)
        logging.info(f"Generated {len(chunks)} chunks")
    except Exception as e:
        logging.error(f"Error during chunking: {e}")
        sys.exit(1)
    
    # Evaluate chunks
    try:
        logging.info("Evaluating chunk quality...")
        evaluation = chunker.evaluate_chunks(chunks)
    except Exception as e:
        logging.error(f"Error evaluating chunks: {e}")
        evaluation = {"error": str(e), "chunk_count": len(chunks)}
    
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
            "chunk_type": chunk.metadata.get("chunk_type", "unknown")
        }
        
        # Include section information if available
        if "section_header" in chunk.metadata:
            chunk_dict["section_header"] = chunk.metadata["section_header"]
            chunk_dict["section_level"] = chunk.metadata.get("section_level", 0)
        
        # Include semantic information if available
        if "semantic_chunk_index" in chunk.metadata:
            chunk_dict["semantic_info"] = {
                "index": chunk.metadata["semantic_chunk_index"],
                "total": chunk.metadata["semantic_chunk_count"]
            }
        
        # Include fixed-size information if available
        if "fixed_chunk_index" in chunk.metadata:
            chunk_dict["fixed_info"] = {
                "index": chunk.metadata["fixed_chunk_index"],
                "total": chunk.metadata["fixed_chunk_count"]
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
    print("\nChunking complete!")
    print(f"Processed 1 document")
    print(f"Generated {len(all_chunks)} total chunks")
    
    # Print token statistics
    if "token_counts" in evaluation:
        token_stats = evaluation["token_counts"]
        print(f"Token counts: min={token_stats.get('min', 0)}, "
              f"max={token_stats.get('max', 0)}, "
              f"avg={token_stats.get('avg', 0):.2f}")
    
    # Print semantic coherence if available
    if "semantic_coherence" in evaluation:
        sem_stats = evaluation["semantic_coherence"]
        print(f"Semantic coherence: min={sem_stats.get('min', 0):.2f}, "
              f"max={sem_stats.get('max', 0):.2f}, "
              f"avg={sem_stats.get('avg', 0):.2f}")
    
    # Print chunk type distribution
    if "chunk_types" in evaluation:
        print("Chunk types:")
        for chunk_type, count in evaluation["chunk_types"].items():
            print(f"  {chunk_type}: {count}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 