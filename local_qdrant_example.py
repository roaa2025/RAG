#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example showing how to connect to a Qdrant instance.
This works with:
- Local in-memory Qdrant
- Docker Qdrant container
- Docker Compose setup

To use:
1. Make sure Qdrant is running (in-memory, Docker, or Docker Compose)
2. Run this script: python local_qdrant_example.py
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams

def connect_to_qdrant():
    """
    Connect to a Qdrant instance using environment variables if available,
    otherwise fall back to localhost.
    """
    # Get connection details from environment variables (for Docker Compose)
    # or use defaults (for local development)
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
    
    # Try to connect to the Qdrant server
    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        # Check connection by getting the list of collections
        client.get_collections()
        print(f"✅ Successfully connected to Qdrant at {qdrant_host}:{qdrant_port}")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant at {qdrant_host}:{qdrant_port}: {e}")
        print("Is Qdrant running? Options:")
        print("1. Run with Docker: docker run -p 6333:6333 qdrant/qdrant")
        print("2. Run with Docker Compose: docker-compose up")
        print("3. Run locally: qdrant.exe --storage-path ./qdrant_storage")
        print("4. Use in-memory mode instead (falling back to this)")
        
        # Fall back to in-memory mode
        try:
            print("\nTrying in-memory mode...")
            client = QdrantClient(":memory:")
            print("✅ Connected to in-memory Qdrant")
            return client
        except Exception as e2:
            print(f"❌ In-memory connection also failed: {e2}")
            return None

def run_example():
    """Run a simple example with the Qdrant instance."""
    # Connect to Qdrant
    client = connect_to_qdrant()
    if client is None:
        return
    
    # Initialize the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector_size = model.get_sentence_embedding_dimension()
    
    # Collection name
    collection_name = "docker_test_collection"
    
    # Create or recreate collection
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        print(f"Collection '{collection_name}' already exists, recreating it...")
        client.delete_collection(collection_name)
    
    # Create a fresh collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection '{collection_name}' with vector size {vector_size}")
    
    # Sample data to store
    documents = [
        "Running Qdrant with Docker Compose",
        "Connecting Python applications to Qdrant vector database",
        "Storing vector embeddings for similarity search",
        "Running Qdrant with persistent storage on disk",
        "Vector databases for semantic text search",
        "Embedding models convert text to vectors for search",
        "Containerized applications with Docker and Python",
        "Orchestrating services with Docker Compose"
    ]
    
    # Create embeddings
    embeddings = model.encode(documents)
    
    # Store in Qdrant
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=list(range(len(documents))),
            vectors=embeddings.tolist(),
            payloads=[{"text": doc, "index": i} for i, doc in enumerate(documents)]
        )
    )
    print(f"Added {len(documents)} documents to '{collection_name}'")
    
    # Search example
    queries = [
        "How to run Qdrant with Docker",
        "Using vector databases for search",
        "Containerization with Docker"
    ]
    
    for query in queries:
        print(f"\nSearch query: '{query}'")
        query_embedding = model.encode(query)
        
        # Execute the search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=3
        )
        
        # Print results
        print(f"Top {len(search_results)} results:")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. [{result.score:.4f}] {result.payload['text']}")

if __name__ == "__main__":
    run_example() 