#!/usr/bin/env python
"""
Test script to verify Qdrant storage and search functionality.
"""

from app.embedding_service import EmbeddingService
import json

def test_qdrant():
    print("Initializing embedding service...")
    service = EmbeddingService()
    
    # Load embeddings from file
    print("\nLoading embeddings from file...")
    with open("output/document_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Store embeddings in Qdrant
    print("\nStoring embeddings in Qdrant...")
    result = service.load_and_store_embeddings(
        json_file_path="output/document_embeddings.json",
        collection_name="document_embeddings"
    )
    
    print(f"\nStorage result: {result}")
    
    # Test search
    print("\nTesting search functionality...")
    test_queries = [
        "data warehouse",
        "GIS technology",
        "data analysis"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = service.search_similar(query, k=3)
        
        if not results:
            print("No results found.")
            continue
            
        print(f"\nFound {len(results)} results:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content: {doc.page_content[:150]}..." if len(doc.page_content) > 150 else doc.page_content)
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            if 'skills' in doc.metadata:
                print(f"Skills: {doc.metadata['skills']}")

if __name__ == "__main__":
    test_qdrant() 