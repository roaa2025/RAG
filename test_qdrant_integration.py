#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify Qdrant integration for document storage and retrieval.
"""

import os
from dotenv import load_dotenv
from app.embedding_service import EmbeddingService
from app.rag_service import RAGService
from app.main import parse_document

# Load environment variables
load_dotenv()

def test_qdrant_integration():
    """Test Qdrant integration with document processing and retrieval."""
    
    # Test document content
    test_doc = """
    Qdrant is a vector similarity search engine and database.
    It is designed to store and retrieve vector embeddings efficiently.
    Vector databases are essential for semantic search and RAG applications.
    LangChain provides excellent integration with Qdrant for building RAG systems.
    """
    
    # Save test document to a temporary file
    with open("test_doc.txt", "w", encoding="utf-8") as f:
        f.write(test_doc)
    
    print("\n1. Testing Document Processing and Embedding Storage")
    print("==================================================")
    
    try:
        # Process document with embeddings enabled
        config = {
            "embeddings": {
                "enabled": True
            }
        }
        
        result = parse_document(
            file_path="test_doc.txt",
            config=config,
            output_dir="output"
        )
        
        print(f"\nDocument processing complete:")
        print(f"- Document count: {result.get('document_count', 0)}")
        print(f"- Chunk count: {result.get('chunk_count', 0)}")
        print(f"- Embedding count: {result.get('embedding_count', 0)}")
        
        if "embedding_error" in result:
            print(f"\nWarning: Error creating embeddings: {result['embedding_error']}")
            return
        
        print("\n2. Testing Qdrant Retrieval")
        print("==========================")
        
        # Initialize RAG service for retrieval
        rag_service = RAGService()
        
        # Test queries
        test_queries = [
            "What is Qdrant?",
            "How does Qdrant work with vector embeddings?",
            "What is the relationship between Qdrant and LangChain?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 50)
            
            # Get answer and sources
            result = rag_service.query(query, k=2)
            
            print("\nAnswer:")
            print(result["answer"])
            
            print("\nSource Documents:")
            for i, source in enumerate(result["source_documents"], 1):
                print(f"\nSource {i}:")
                print(f"Content: {source['content']}")
                print(f"Score: {source['score']:.4f}")
                print(f"Metadata: {source['metadata']}")
        
        print("\n3. Testing Similarity Search")
        print("===========================")
        
        # Test similarity search
        search_query = "vector database similarity search"
        print(f"\nSearching for: {search_query}")
        
        similar_docs = rag_service.search_similar(search_query, k=2)
        print("\nSimilar Documents:")
        for i, doc in enumerate(similar_docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
        
        print("\n✅ Qdrant integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test file
        if os.path.exists("test_doc.txt"):
            os.remove("test_doc.txt")

if __name__ == "__main__":
    test_qdrant_integration() 