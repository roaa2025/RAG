#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script demonstrates how to use Qdrant with text embeddings
to create a practical vector storage solution for document retrieval.
"""

import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams

# Load environment variables (if using API keys)
load_dotenv()

class QdrantDocumentStore:
    """
    A class to handle document storage and retrieval using Qdrant vector database.
    """
    
    def __init__(self, 
                 use_in_memory=True, 
                 collection_name="documents",
                 host="localhost",
                 port=6333,
                 embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the document store.
        
        Args:
            use_in_memory (bool): If True, use in-memory storage, otherwise use Docker/external
            collection_name (str): Name of the collection in Qdrant
            host (str): Host address for Qdrant (ignored if use_in_memory=True)
            port (int): Port for Qdrant (ignored if use_in_memory=True)
            embedding_model (str): Name of the sentence-transformers model to use
        """
        # Initialize the embedding model
        self.model = SentenceTransformer(embedding_model)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        self.collection_name = collection_name
        
        # Get connection details from environment variables if available
        env_host = os.environ.get("QDRANT_HOST")
        env_port = os.environ.get("QDRANT_PORT")
        
        if env_host:
            host = env_host
        if env_port:
            port = int(env_port)
        
        # Initialize Qdrant client
        if use_in_memory:
            self.client = QdrantClient(":memory:")
            print(f"Using in-memory Qdrant storage")
        else:
            self.client = QdrantClient(host=host, port=port)
            print(f"Connected to Qdrant at {host}:{port}")
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create the collection if it doesn't already exist."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            print(f"Created collection '{self.collection_name}' with vector size {self.vector_size}")
        else:
            print(f"Using existing collection '{self.collection_name}'")
    
    def add_documents(self, documents, metadatas=None):
        """
        Add documents to the vector store.
        
        Args:
            documents (list): List of text documents
            metadatas (list, optional): List of metadata dictionaries for each document
        
        Returns:
            list: List of generated IDs
        """
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]
        
        # Generate embeddings for documents
        embeddings = self.model.encode(documents)
        
        # Generate IDs for documents
        collection_count = self._get_collection_count()
        ids = list(range(
            collection_count, 
            collection_count + len(documents)
        ))
        
        # Create payloads
        payloads = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            payload = {"text": doc}
            payload.update(meta)
            payloads.append(payload)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings.tolist(),
                payloads=payloads
            )
        )
        
        print(f"Added {len(documents)} documents to collection '{self.collection_name}'")
        return ids
    
    def _get_collection_count(self):
        """Get the current count of vectors in the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            if hasattr(collection_info, 'vectors_count') and collection_info.vectors_count is not None:
                return collection_info.vectors_count
            return 0  # Return 0 if vectors_count is None or not available
        except Exception as e:
            print(f"Warning: Could not get collection count: {e}")
            return 0  # Return 0 on any error
    
    def search(self, query, limit=5):
        """
        Search for documents similar to the query.
        
        Args:
            query (str): The search query
            limit (int): Maximum number of results to return
        
        Returns:
            list: List of search results with text and metadata
        """
        # Generate embedding for the query
        query_embedding = self.model.encode(query)
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        # Format results
        results = []
        for result in search_results:
            item = {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
            }
            
            # Add all other metadata from payload
            for key, value in result.payload.items():
                if key != "text":
                    item[key] = value
            
            results.append(item)
        
        return results

def run_example():
    """Run a demonstration of the QdrantDocumentStore."""
    # Sample documents
    documents = [
        "Qdrant is a vector similarity search engine and database.",
        "Vector databases store vector embeddings and enable efficient similarity search.",
        "Sentence transformers are used to convert text into embeddings.",
        "Python is a high-level programming language known for its readability.",
        "Docker provides a way to package applications into containers.",
        "Machine learning models can be deployed using various techniques.",
        "Natural language processing helps computers understand human language.",
        "Document retrieval systems help find relevant information quickly.",
        "Docker Compose orchestrates multiple containers as a single application.",
        "Containerization makes applications portable across different environments."
    ]
    
    # Sample metadata
    metadatas = [
        {"category": "databases", "source": "documentation"},
        {"category": "databases", "source": "article"},
        {"category": "nlp", "source": "tutorial"},
        {"category": "programming", "source": "book"},
        {"category": "deployment", "source": "documentation"},
        {"category": "machine_learning", "source": "article"},
        {"category": "nlp", "source": "book"},
        {"category": "information_retrieval", "source": "paper"},
        {"category": "deployment", "source": "tutorial"},
        {"category": "deployment", "source": "article"}
    ]
    
    print("=== Qdrant Document Store Example ===")
    
    # Determine the runtime environment
    is_docker = os.environ.get("QDRANT_HOST") is not None
    
    # Create document store (in-memory for local, remote for Docker)
    print("\n1. INITIALIZING DOCUMENT STORE")
    print("-----------------------------")
    
    if is_docker:
        print("Detected Docker environment, connecting to Qdrant service...")
        # When running in Docker Compose, the host is the service name
        doc_store = QdrantDocumentStore(use_in_memory=False, host=os.environ.get("QDRANT_HOST", "qdrant"))
    else:
        print("Using in-memory Qdrant for local testing...")
        doc_store = QdrantDocumentStore(use_in_memory=True)
    
    # Add documents
    print("\n2. ADDING DOCUMENTS")
    print("------------------")
    doc_store.add_documents(documents, metadatas)
    
    # Search examples
    print("\n3. SEARCHING FOR DOCUMENTS")
    print("-------------------------")
    
    search_queries = [
        "vector database for similarity search",
        "machine learning deployment",
        "natural language understanding",
        "programming languages",
        "docker containerization"
    ]
    
    for query in search_queries:
        print(f"\nQuery: '{query}'")
        results = doc_store.search(query, limit=3)
        
        print(f"Top {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. [{result['score']:.4f}] {result['text']} (Category: {result.get('category', 'unknown')})")
    
    print("\n=== Example Completed Successfully ===")
    print("To use in your own code:")
    print("1. Create a store: store = QdrantDocumentStore()")
    print("2. Add documents: store.add_documents(docs, metadatas)")
    print("3. Search: results = store.search('your query')")

if __name__ == "__main__":
    run_example() 