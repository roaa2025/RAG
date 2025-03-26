#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script creates a Qdrant collection named 'document_embeddings' with:
- Dynamically determined vector size (based on the embedding model)
- Cosine similarity as the distance metric

The script supports:
1. Local in-memory Qdrant (for testing)
2. Local persistent Qdrant 
3. Remote Qdrant (by providing host/port)

Usage:
    python create_document_embeddings_collection.py --mode=local|memory|remote
    
Options:
    --mode: Where to create the collection (default: local)
    --host: Qdrant host for remote mode (default: localhost)
    --port: Qdrant port (default: 6333)
    --model: Embedding model to use (default: text-embedding-3-small)
"""

import os
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, Distance
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_embedding_dimension(model_name):
    """Get the vector dimension for a given embedding model."""
    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create a sample embedding to determine the vector size
    sample_text = "This is a sample text to determine the embedding dimension."
    sample_embedding = embeddings.embed_query(sample_text)
    
    # Return the dimension
    dimension = len(sample_embedding)
    return dimension

def create_document_embeddings_collection(client, model_name="text-embedding-3-small"):
    """
    Create a collection named 'document_embeddings' with:
    - Vector size matching the embedding model dimension
    - Cosine similarity as distance metric
    
    Args:
        client: QdrantClient instance
        model_name: Name of the embedding model to use
    
    Returns:
        Tuple of (collection_name, vector_size)
    """
    collection_name = "document_embeddings"
    
    # Get the vector size from the embedding model
    print(f"Getting vector dimension for model '{model_name}'...")
    vector_size = get_embedding_dimension(model_name)
    print(f"Determined vector size: {vector_size}")
    
    # Check if collection already exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        print(f"Collection '{collection_name}' already exists.")
        # Optionally, you could recreate it:
        # client.delete_collection(collection_name=collection_name)
        # print(f"Deleted existing collection '{collection_name}'")
    else:
        # Create the collection with the determined vector size
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
        )
        print(f"Created collection '{collection_name}' with vector size {vector_size}")
    
    return collection_name, vector_size

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a Qdrant collection for document embeddings"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="local",
        choices=["memory", "local", "remote"],
        help="Where to create the collection (memory, local, or remote)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="Qdrant host address (for remote mode)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6333,
        help="Qdrant port number"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="text-embedding-3-small",
        help="Embedding model to use"
    )
    
    args = parser.parse_args()
    
    # Initialize Qdrant client based on mode
    if args.mode == "memory":
        print("Initializing in-memory Qdrant client...")
        client = QdrantClient(":memory:")
    elif args.mode == "local":
        print("Initializing local Qdrant client...")
        client = QdrantClient("localhost", port=args.port)
    else:  # remote
        print(f"Connecting to remote Qdrant at {args.host}:{args.port}...")
        client = QdrantClient(
            host=args.host,
            port=args.port
        )
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    # Create the collection
    collection_name, vector_size = create_document_embeddings_collection(
        client,
        model_name=args.model
    )
    
    # Verify the collection was created
    print("\nVerifying collection...")
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"Collection Name: {collection_info.name}")
    print(f"Vector Size: {collection_info.config.params.vectors.size}")
    print(f"Distance Function: {collection_info.config.params.vectors.distance}")
    
    print("\nCollection 'document_embeddings' successfully created and configured!")
    print("\nTo use this collection in your code:")
    print("```python")
    print("from qdrant_client import QdrantClient")
    if args.mode == "memory":
        print("client = QdrantClient(':memory:')")
    elif args.mode == "local":
        print(f"client = QdrantClient('localhost', port={args.port})")
    else:  # remote
        print(f"client = QdrantClient(host='{args.host}', port={args.port})")
    print("```")

if __name__ == "__main__":
    main() 