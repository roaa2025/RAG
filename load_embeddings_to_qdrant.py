#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script loads text embeddings from output/document_embeddings.json 
and stores them in a Qdrant collection with structured metadata.

Features:
- Loads pre-generated embeddings from JSON file
- Stores embeddings in Qdrant collection 'document_embeddings'
- Uses optimized batch insertion for performance
- Stores structured metadata (id, vector, payload with extracted text)
- Provides progress reporting and timing statistics

Usage:
    python load_embeddings_to_qdrant.py [--file FILE] [--collection COLLECTION] [--batch_size BATCH_SIZE]
    
Options:
    --file: Path to JSON file with embeddings (default: output/document_embeddings.json)
    --collection: Name of the Qdrant collection (default: document_embeddings)
    --batch_size: Number of embeddings to insert in a single batch (default: 100)
    --host: Qdrant host (default: localhost)
    --port: Qdrant port (default: 6333)
"""

import os
import argparse
import json
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from app.embedding_service import EmbeddingService

# Load environment variables from .env file
load_dotenv()

def main():
    """Main function to load embeddings and store them in Qdrant."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Load text embeddings from JSON and store them in Qdrant"
    )
    parser.add_argument(
        "--file", 
        type=str, 
        default="output/document_embeddings.json",
        help="Path to the JSON file with embeddings"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="document_embeddings",
        help="Name of the Qdrant collection"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=100,
        help="Number of embeddings to insert in a single batch"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="Qdrant host"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6333,
        help="Qdrant port"
    )
    
    args = parser.parse_args()
    
    # Verify the JSON file exists
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Embeddings file not found: {args.file}")
    
    # Initialize Qdrant client
    print(f"Connecting to Qdrant at {args.host}:{args.port}...")
    qdrant_client = QdrantClient(args.host, port=args.port)
    
    # Initialize the embedding service with the Qdrant client
    embedding_service = EmbeddingService(qdrant_client=qdrant_client)
    
    # Load embeddings from JSON and store them in Qdrant
    try:
        stats = embedding_service.load_and_store_embeddings(
            json_file_path=args.file,
            collection_name=args.collection,
            batch_size=args.batch_size
        )
        
        # Print summary statistics
        print("\n=== Summary ===")
        print(f"Collection: {stats['collection_name']}")
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"Embeddings inserted: {stats['embeddings_inserted']}")
        print(f"Elapsed time: {stats['elapsed_seconds']:.2f} seconds")
        print(f"Batch size: {stats['batch_size_used']}")
        
        if stats['embeddings_inserted'] > 0:
            insert_rate = stats['embeddings_inserted'] / stats['elapsed_seconds']
            print(f"Insertion rate: {insert_rate:.2f} embeddings/second")
        
        # Print usage example
        print("\nTo search this collection in your Python code:")
        print("```python")
        print("from qdrant_client import QdrantClient")
        print(f"client = QdrantClient('{args.host}', port={args.port})")
        print(f"results = client.search(")
        print(f"    collection_name='{args.collection}',")
        print(f"    query_vector=your_query_embedding,  # Replace with your query vector")
        print(f"    limit=10")
        print(f")")
        print("```")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 