#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script demonstrates how to set up Qdrant vector database in two ways:
1. In-memory mode (:memory:) for testing
2. As a persistent instance using Docker

Prerequisites:
- Install qdrant-client: pip install qdrant-client
- For Docker mode: Docker installed and running
  (docker run -p 6333:6333 qdrant/qdrant)
"""

import os
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams

def setup_in_memory_qdrant():
    """Set up an in-memory Qdrant instance."""
    print("Setting up in-memory Qdrant instance...")
    
    # Initialize in-memory Qdrant client
    client = QdrantClient(":memory:")
    
    # Create a test collection
    collection_name = "test_collection"
    vector_size = 768  # Common size for embeddings, adjust as needed
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    
    print(f"Created collection '{collection_name}' with vector size {vector_size}")
    
    # Test with sample vectors
    test_in_memory_client(client, collection_name, vector_size)
    
    return client

def setup_docker_qdrant():
    """Set up a connection to a Qdrant instance running in Docker."""
    print("\nConnecting to Qdrant running in Docker...")
    print("Note: Make sure you've started Qdrant with:")
    print("docker run -p 6333:6333 qdrant/qdrant")
    
    # Initialize Qdrant client pointing to Docker instance
    try:
        client = QdrantClient("localhost", port=6333)
        
        # Create a test collection
        collection_name = "docker_test_collection"
        vector_size = 768  # Common size for embeddings, adjust as needed
        
        # Check if collection exists and create if it doesn't
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Created collection '{collection_name}' with vector size {vector_size}")
        else:
            print(f"Collection '{collection_name}' already exists")
        
        # Test with sample vectors
        test_docker_client(client, collection_name, vector_size)
        
        return client
    except Exception as e:
        print(f"Failed to connect to Docker Qdrant instance: {e}")
        print("Is Docker running with Qdrant container started?")
        return None

def test_in_memory_client(client, collection_name, vector_size):
    """Test the in-memory Qdrant client with sample data."""
    # Create sample vectors (random embeddings)
    sample_vectors = [np.random.rand(vector_size).tolist() for _ in range(5)]
    sample_payloads = [{"metadata": f"Document {i}", "text": f"Sample text {i}"} for i in range(5)]
    
    # Upload vectors
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=list(range(5)),
            vectors=sample_vectors,
            payloads=sample_payloads
        )
    )
    
    print(f"Uploaded 5 sample vectors to '{collection_name}'")
    
    # Verify data was uploaded by searching
    query_vector = np.random.rand(vector_size).tolist()
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )
    
    print(f"Successfully retrieved {len(search_result)} results from search")
    
    # Show some details about the first result
    if search_result:
        print(f"First result ID: {search_result[0].id}, score: {search_result[0].score:.4f}")
        print(f"Metadata: {search_result[0].payload['metadata']}")
    
    print("In-memory Qdrant test completed successfully")

def test_docker_client(client, collection_name, vector_size):
    """Test the Docker Qdrant client with sample data."""
    try:
        # Create sample vectors (random embeddings)
        sample_vectors = [np.random.rand(vector_size).tolist() for _ in range(5)]
        sample_payloads = [{"metadata": f"Docker Document {i}", "text": f"Docker Sample text {i}"} for i in range(5)]
        
        # Upload vectors
        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=list(range(100, 105)),  # Use different IDs for clarity
                vectors=sample_vectors,
                payloads=sample_payloads
            )
        )
        
        print(f"Uploaded 5 sample vectors to Docker '{collection_name}'")
        
        # Verify data was uploaded by searching
        query_vector = np.random.rand(vector_size).tolist()
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )
        
        print(f"Successfully retrieved {len(search_result)} results from search")
        
        # Show some details about the first result
        if search_result:
            print(f"First result ID: {search_result[0].id}, score: {search_result[0].score:.4f}")
            print(f"Metadata: {search_result[0].payload['metadata']}")
        
        print("Docker Qdrant test completed successfully")
    except Exception as e:
        print(f"Error testing Docker Qdrant client: {e}")

if __name__ == "__main__":
    print("=== Qdrant Vector Database Setup ===")
    
    # Setup in-memory Qdrant
    print("\n1. TESTING IN-MEMORY QDRANT")
    print("---------------------------")
    in_memory_client = setup_in_memory_qdrant()
    
    # Setup Docker Qdrant
    print("\n2. TESTING DOCKER QDRANT")
    print("----------------------")
    print("Attempting to connect to Qdrant running in Docker...")
    docker_client = setup_docker_qdrant()
    
    if docker_client is None:
        print("\nDocker connection failed. To run Qdrant in Docker:")
        print("1. Make sure Docker is installed and running")
        print("2. Execute: docker run -p 6333:6333 qdrant/qdrant")
        print("3. Run this script again")
    
    print("\n=== Summary ===")
    print("✅ In-memory Qdrant setup and test completed")
    print(f"{'✅' if docker_client else '❌'} Docker Qdrant setup and test")
    
    print("\nTo use Qdrant in your application:")
    print("- For testing: client = QdrantClient(':memory:')")
    print("- For production: client = QdrantClient('localhost', port=6333)  # Docker")
    print("\nSee documentation for more options: https://qdrant.tech/documentation/") 