#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.rag_service import RAGService
import os
from dotenv import load_dotenv

def print_menu():
    print("\n=== RAG Service Test Menu ===")
    print("1. Search for similar documents")
    print("2. Ask a question")
    print("3. Search with similarity scores")
    print("4. Get document by ID")
    print("5. Exit")
    print("===========================")

def print_document(doc, include_score=False):
    print("\n--- Document ---")
    print(f"Content: {doc['content'][:200]}...")  # Show first 200 chars
    print(f"Metadata: {doc['metadata']}")
    if include_score:
        print(f"Score: {doc['score']}")
    print("---------------")

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize RAG service
        rag = RAGService()
        print("RAG service initialized successfully!")
        
        while True:
            print_menu()
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                query = input("\nEnter your search query: ")
                results = rag.search_similar(query)
                print(f"\nFound {len(results)} similar documents:")
                for doc in results:
                    print_document({"content": doc.page_content, "metadata": doc.metadata})
                    
            elif choice == "2":
                question = input("\nEnter your question: ")
                response = rag.query(question)
                
                if "error" in response:
                    print(f"\nError: {response['error']}")
                    continue
                    
                print("\nAnswer:", response["answer"])
                print("\nSource Documents:")
                for doc in response["source_documents"]:
                    print_document(doc)
                    
            elif choice == "3":
                query = input("\nEnter your search query: ")
                results = rag.search_with_score(query)
                print(f"\nFound {len(results)} similar documents:")
                for doc, score in results:
                    print_document({"content": doc.page_content, "metadata": doc.metadata, "score": score}, include_score=True)
                    
            elif choice == "4":
                doc_id = input("\nEnter document ID: ")
                doc = rag.get_document_by_id(doc_id)
                if doc:
                    print_document({"content": doc.page_content, "metadata": doc.metadata})
                else:
                    print("Document not found")
                    
            elif choice == "5":
                print("\nGoodbye!")
                break
                
            else:
                print("\nInvalid choice. Please try again.")
                
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure Qdrant is running on localhost:6333")

if __name__ == "__main__":
    main() 