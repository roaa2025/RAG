#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process existing parsed documents:
1. Load parsed documents from JSON file (from previous pipeline execution)
2. Extract and process chunks
3. Create embeddings for document chunks
4. Store in Qdrant vector store

This script allows you to skip the document extraction and initial parsing 
steps if you already have parsed documents in JSON format.

Example usage:
    python process_existing_documents.py output/parsed_documents.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from app.embedding_service import EmbeddingService
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def extract_chunks_from_parsed_documents(parsed_documents_json):
    """
    Extract chunks from the parsed documents JSON structure.
    
    Args:
        parsed_documents_json: Loaded JSON data from parsed_documents.json
        
    Returns:
        List of document chunks with their metadata
    """
    chunks = []
    
    # Handle different document structure formats
    if "processed_documents" in parsed_documents_json:
        # Format from basic structurer
        documents = parsed_documents_json["processed_documents"]
        for doc in documents:
            if "chunks" in doc:
                # Document has chunks array
                for i, chunk in enumerate(doc["chunks"]):
                    chunks.append({
                        "content": chunk.get("content", ""),
                        "metadata": {
                            "doc_id": doc.get("id", f"doc_{i}"),
                            "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                            "document_title": doc.get("title", ""),
                            "content_type": chunk.get("content_type", "text"),
                            **chunk.get("metadata", {})
                        }
                    })
            else:
                # Treat the whole document as one chunk
                chunks.append({
                    "content": doc.get("raw_text", ""),
                    "metadata": {
                        "doc_id": doc.get("id", f"doc_{len(chunks)}"),
                        "chunk_id": f"chunk_{len(chunks)}",
                        "document_title": doc.get("title", ""),
                        "content_type": doc.get("content_type", "text"),
                        **doc.get("metadata", {})
                    }
                })
    elif "all_chunks" in parsed_documents_json:
        # Format from advanced structurer
        for i, chunk in enumerate(parsed_documents_json["all_chunks"]):
            chunks.append({
                "content": chunk.get("content", ""),
                "metadata": {
                    "doc_id": chunk.get("doc_id", f"doc_{i}"),
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "document_title": chunk.get("title", ""),
                    "content_type": chunk.get("content_type", "text"),
                    "chunk_type": chunk.get("chunk_type", "unknown"),
                    **chunk.get("metadata", {})
                }
            })
    elif "structured_chunks" in parsed_documents_json:
        # Another possible format
        for i, chunk in enumerate(parsed_documents_json["structured_chunks"]):
            chunks.append({
                "content": chunk.get("content", ""),
                "metadata": {
                    "doc_id": chunk.get("doc_id", f"doc_{i}"),
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "document_title": chunk.get("title", ""),
                    "content_type": chunk.get("content_type", "text"),
                    **chunk.get("metadata", {})
                }
            })
    
    # If no chunks were found, try to process as a simple document array
    if not chunks and isinstance(parsed_documents_json, list):
        for i, doc in enumerate(parsed_documents_json):
            chunks.append({
                "content": doc.get("content", doc.get("text", doc.get("page_content", ""))),
                "metadata": {
                    "doc_id": doc.get("id", f"doc_{i}"),
                    "chunk_id": f"chunk_{i}",
                    **doc.get("metadata", {})
                }
            })
    
    return chunks

def create_document_objects(chunks):
    """
    Create Langchain Document objects from chunks.
    
    Args:
        chunks: List of chunks with content and metadata
        
    Returns:
        List of Document objects
    """
    documents = []
    
    for chunk in chunks:
        if not chunk.get("content", ""):
            continue
            
        # Create Document object
        doc = Document(
            page_content=chunk["content"],
            metadata=chunk["metadata"]
        )
        documents.append(doc)
    
    return documents

def process_parsed_documents(input_json_path, output_dir="output", collection_name="document_embeddings", force=False):
    """
    Process existing parsed documents to create embeddings and store in Qdrant.
    
    Args:
        input_json_path: Path to the parsed documents JSON file
        output_dir: Directory to save output files
        collection_name: Name of the Qdrant collection
        force: Force reprocessing even if embeddings exist
        
    Returns:
        Dictionary with processing results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parsed documents JSON
    logger.info(f"Loading parsed documents from {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        parsed_documents = json.load(f)
    
    # Check if this is a pipeline result file and extract parsed_documents if needed
    if "parsed_documents" in parsed_documents and isinstance(parsed_documents["parsed_documents"], (dict, list)):
        parsed_documents = parsed_documents["parsed_documents"]
    
    # Extract chunks from parsed documents
    logger.info("Extracting document chunks")
    chunks = extract_chunks_from_parsed_documents(parsed_documents)
    logger.info(f"Extracted {len(chunks)} chunks from parsed documents")
    
    # Check if chunks were extracted
    if not chunks:
        logger.error("No document chunks found in the parsed documents")
        return {"error": "No document chunks found", "chunk_count": 0}
    
    # Create Document objects
    logger.info("Creating Document objects")
    documents = create_document_objects(chunks)
    logger.info(f"Created {len(documents)} Document objects")
    
    # Initialize Qdrant client
    logger.info("Connecting to Qdrant")
    qdrant_client = QdrantClient("localhost", port=6333)
    
    # Initialize embedding service with Qdrant client
    logger.info("Initializing embedding service")
    embedding_service = EmbeddingService(qdrant_client=qdrant_client)
    
    # Create embeddings for documents
    logger.info("Creating embeddings for document chunks")
    embeddings_result = embedding_service.process_documents_to_embeddings(documents)
    
    # Save embeddings to file
    embeddings_path = os.path.join(output_dir, "document_embeddings.json")
    logger.info(f"Saving embeddings to {embeddings_path}")
    embedding_service.save_embeddings(embeddings_result, embeddings_path)
    
    # Store embeddings in Qdrant
    logger.info(f"Storing embeddings in Qdrant collection '{collection_name}'")
    qdrant_results = embedding_service.load_and_store_embeddings(
        json_file_path=embeddings_path,
        collection_name=collection_name,
        batch_size=100
    )
    
    # Create result summary
    result = {
        "input_file": input_json_path,
        "output_dir": output_dir,
        "chunk_count": len(chunks),
        "document_count": len(documents),
        "embedding_count": len(embeddings_result.get("processed_chunks", [])),
        "embeddings_path": embeddings_path,
        "qdrant_collection": collection_name,
        "qdrant_storage": {
            "embeddings_stored": qdrant_results.get("embeddings_inserted", 0),
            "elapsed_seconds": qdrant_results.get("elapsed_seconds", 0),
        }
    }
    
    # Save result to file
    result_path = os.path.join(output_dir, "embedding_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {result_path}")
    
    return result

def main():
    """Process parsed documents and create embeddings."""
    parser = argparse.ArgumentParser(
        description="Process existing parsed documents and create embeddings"
    )
    parser.add_argument(
        "json_file", 
        type=str,
        help="Path to the parsed documents JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="document_embeddings",
        help="Name of the Qdrant collection"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force reprocessing even if embeddings exist"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input file exists
    if not os.path.exists(args.json_file):
        logger.error(f"Input file not found: {args.json_file}")
        return 1
    
    try:
        # Process parsed documents
        print(f"\n🚀 Processing parsed documents from: {args.json_file}")
        print(f"💾 Output directory: {args.output_dir}")
        
        result = process_parsed_documents(
            input_json_path=args.json_file,
            output_dir=args.output_dir,
            collection_name=args.collection,
            force=args.force
        )
        
        # Print summary
        print("\n✅ Processing completed successfully!")
        print(f"📄 Processed {result['document_count']} documents")
        print(f"🧩 Processed {result['chunk_count']} chunks")
        print(f"🔢 Created {result['embedding_count']} embeddings")
        
        # Print Qdrant storage information
        qdrant_info = result.get("qdrant_storage", {})
        print(f"💾 Stored {qdrant_info.get('embeddings_stored', 0)} embeddings in Qdrant")
        print(f"⏱️ Vector storage completed in {qdrant_info.get('elapsed_seconds', 0):.2f} seconds")
        
        # Print output file locations
        print("\n📂 Result files:")
        print(f"  - Embeddings: {result['embeddings_path']}")
        result_path = os.path.join(args.output_dir, "embedding_results.json")
        print(f"  - Results: {result_path}")
        
        # Print search instructions
        print("\n🔍 To search the stored documents:")
        print("1. Run a search using memory_search.py or search_documents.py")
        print("2. Example: python memory_search.py --query \"your query here\"")
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 