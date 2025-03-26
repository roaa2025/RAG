#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple script that sets up an in-memory Qdrant collection with sample data
and performs a semantic search, all in one step.

This script doesn't require a running Qdrant server - everything happens in memory.

Usage:
    python memory_search.py "your search query"
    
Example:
    python memory_search.py "vector database"
"""

import sys
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_community.vectorstores import Qdrant

# Load environment variables
load_dotenv()

def main():
    # Check if a search query was provided
    if len(sys.argv) < 2:
        print("Error: You must provide a search query.")
        print("Usage: python memory_search.py \"your search query\"")
        return 1
    
    # Extract search query
    query = sys.argv[1]
    
    # Get optional top_k
    top_k = 3
    with_scores = False
    
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--top_k" and i + 1 < len(sys.argv):
            try:
                top_k = int(sys.argv[i + 1])
            except ValueError:
                print(f"Error: --top_k requires an integer value")
                return 1
        elif sys.argv[i] == "--with_scores":
            with_scores = True
    
    print("\n===================================")
    print("🔍 In-Memory Semantic Search Demo 🔍")
    print("===================================")
    
    print("\nSetting up in-memory Qdrant...")
    
    # Initialize in-memory Qdrant client
    client = QdrantClient(":memory:")
    
    # Initialize OpenAI embeddings
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Sample documents for demo
    sample_docs = [
        "Semantic search is a search technique that understands the context of search queries.",
        "Vector embeddings are numerical representations of text that capture semantic meaning.",
        "Qdrant is a vector database that allows for efficient similarity search.",
        "LangChain provides tools for working with language models and vector stores.",
        "Document retrieval systems can use embeddings to find relevant information.",
    ]
    
    # Create embeddings for sample docs
    print("Creating embeddings for sample documents...")
    sample_embeddings = embedding_model.embed_documents(sample_docs)
    vector_size = len(sample_embeddings[0])
    
    # Create collection
    collection_name = "document_embeddings"
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    
    # Insert sample documents
    points = []
    for i, (doc, embedding) in enumerate(zip(sample_docs, sample_embeddings)):
        # Qdrant payload structure matters for LangChain compatibility
        # LangChain expects 'page_content' for the text and 'metadata' for additional data
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "page_content": doc,  # This key is required by LangChain
                "metadata": {
                    "doc_id": f"sample_{i}",
                    "source": "sample_data"
                }
            }
        ))
    
    print(f"Inserting {len(points)} sample documents...")
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    # Initialize LangChain vectorstore with the correct content payload key
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_model,
        content_payload_key="page_content"  # Match the key used in the payload
    )
    
    # Perform search
    print(f"\n🔎 Searching for: \"{query}\"")
    print(f"   Returning top {top_k} results")
    
    if with_scores:
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        
        if not results:
            print("\n❌ No results found.")
            return 0
            
        print(f"\n✅ Found {len(results)} results:\n")
        
        for i, (doc, score) in enumerate(results):
            print(f"Result #{i+1} [Similarity: {score:.4f}]")
            print("=" * 50)
            print(f"📄 Content: {doc.page_content}")
            print("\n🏷️  Metadata:")
            
            for meta_key, meta_value in doc.metadata.items():
                print(f"  • {meta_key}: {meta_value}")
            print()
    else:
        results = vectorstore.similarity_search(query, k=top_k)
        
        if not results:
            print("\n❌ No results found.")
            return 0
            
        print(f"\n✅ Found {len(results)} results:\n")
        
        for i, doc in enumerate(results):
            print(f"Result #{i+1}")
            print("=" * 50)
            print(f"📄 Content: {doc.page_content}")
            print("\n🏷️  Metadata:")
            
            for meta_key, meta_value in doc.metadata.items():
                print(f"  • {meta_key}: {meta_value}")
            print()
    
    print("\nNote: This was an in-memory demo. The data only exists during this script's execution.")
    print("To learn how to use persistent storage, see the QDRANT_SEARCH_README.md file.")
    
    return 0

if __name__ == "__main__":
    exit(main()) 