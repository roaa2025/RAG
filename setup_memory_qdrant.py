#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script sets up an in-memory Qdrant collection with sample data for testing.
No external Qdrant server is required - everything runs in memory.

It creates:
1. An in-memory Qdrant instance
2. A collection called 'document_embeddings' with appropriate vector size
3. Sample documents with embeddings for testing

Usage:
    python setup_memory_qdrant.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np

# Load environment variables
load_dotenv()

def create_sample_embeddings(texts, embeddings_model):
    """Create embeddings for sample texts."""
    return embeddings_model.embed_documents(texts)

def main():
    print("Setting up in-memory Qdrant for testing...")
    
    # Initialize in-memory Qdrant client
    client = QdrantClient(":memory:")
    
    # Initialize the embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create sample texts
    sample_texts = [
        "Semantic search is a search technique that understands the context of search queries.",
        "Vector embeddings are numerical representations of text that capture semantic meaning.",
        "Qdrant is a vector database that allows for efficient similarity search.",
        "LangChain provides tools for working with language models and vector stores.",
        "Document retrieval systems can use embeddings to find relevant information.",
    ]
    
    # Create embeddings for the sample texts
    print("Creating embeddings for sample texts...")
    embeddings = create_sample_embeddings(sample_texts, embedding_model)
    vector_size = len(embeddings[0])
    print(f"Sample embedding dimension: {vector_size}")
    
    # Create collection
    collection_name = "document_embeddings"
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        ),
    )
    
    # Prepare points for insertion
    points = []
    for i, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
        # Metadata with source and doc_id
        metadata = {
            "doc_id": f"sample_{i}",
            "source": "sample_data"
        }
        
        # Create point
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "text": text,
                "metadata": metadata
            }
        ))
    
    # Insert points
    print(f"Inserting {len(points)} sample documents...")
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    # Verify the collection
    collection_info = client.get_collection(collection_name=collection_name)
    count = client.count(collection_name=collection_name).count
    
    print("\nIn-memory Qdrant setup complete!")
    print(f"Collection: {collection_name}")
    print(f"Vector dimension: {vector_size}")
    print(f"Documents: {count}")
    print(f"Distance: {collection_info.config.params.vectors.distance}")
    
    print("\nYou can now use this in-memory collection for testing with:")
    print("```python")
    print("from app.qdrant_search import QdrantSearchService")
    print("")
    print("# Initialize with in-memory mode")
    print("search_service = QdrantSearchService(")
    print("    host=\":memory:\",")
    print("    collection_name=\"document_embeddings\"")
    print(")")
    print("")
    print("# Perform queries")
    print("results = search_service.similarity_search(\"your query here\")")
    print("```")
    
    print("\nOr from the command line:")
    print("python search_documents.py \"your query here\" --host=\":memory:\"")

if __name__ == "__main__":
    main() 