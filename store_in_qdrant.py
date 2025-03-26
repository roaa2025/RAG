#!/usr/bin/env python
"""
Script to store document embeddings and skills in Qdrant vector database.
This takes the embeddings JSON file created by run_docling.py --create-embeddings
and stores the vectors in Qdrant for semantic search.
"""

import os
import argparse
import json
from dotenv import load_dotenv
from tqdm import tqdm
from app.embedding_service import EmbeddingService

def store_embeddings_in_qdrant(embeddings_file, collection_name="document_embeddings", batch_size=100):
    """
    Store document embeddings in Qdrant vector database.
    
    Args:
        embeddings_file: Path to the embeddings JSON file
        collection_name: Name of the Qdrant collection
        batch_size: Number of embeddings to insert in a single batch
    
    Returns:
        Dictionary with storage statistics
    """
    print(f"Initializing embedding service...")
    embedding_service = EmbeddingService()
    
    print(f"Loading embeddings from: {embeddings_file}")
    try:
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total_chunks = len(data.get("processed_chunks", []))
        if total_chunks == 0:
            print("No embeddings found in the file.")
            return {"error": "No embeddings found"}
            
        print(f"Found {total_chunks} embeddings to store")
        
        # Store in Qdrant
        print(f"Storing embeddings in Qdrant collection: {collection_name}")
        result = embedding_service.load_and_store_embeddings(
            json_file_path=embeddings_file,
            collection_name=collection_name,
            batch_size=batch_size
        )
        
        print(f"\nSuccess! Stored {result['embeddings_inserted']} embeddings in Qdrant")
        print(f"Collection name: {collection_name}")
        print(f"Time taken: {result['elapsed_seconds']:.2f} seconds")
        
        return result
        
    except FileNotFoundError:
        print(f"Error: Embeddings file not found at {embeddings_file}")
        return {"error": "File not found"}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in embeddings file")
        return {"error": "Invalid JSON"}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

def verify_qdrant_storage(collection_name="document_embeddings", query="data warehouse"):
    """
    Verify that embeddings were stored correctly by performing a test search.
    
    Args:
        collection_name: Name of the Qdrant collection
        query: Test query to search for
        
    Returns:
        True if search was successful, False otherwise
    """
    print(f"\nVerifying storage with test query: '{query}'")
    try:
        embedding_service = EmbeddingService()
        results = embedding_service.search_similar(query, k=3)
        
        if not results:
            print("No results found. Storage may not be working correctly.")
            return False
            
        print(f"\nFound {len(results)} results:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            
            # Print a snippet of the content
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"Content: {content_preview}")
            
            # Print skills if available
            skills = doc.metadata.get('skills', [])
            if skills:
                if isinstance(skills, list):
                    # Handle both list of strings and list of dicts
                    if skills and isinstance(skills[0], dict) and 'name' in skills[0]:
                        skill_names = [s['name'] for s in skills]
                        print(f"Skills: {', '.join(skill_names[:5])}" + ("..." if len(skill_names) > 5 else ""))
                    else:
                        print(f"Skills: {', '.join(skills[:5])}" + ("..." if len(skills) > 5 else ""))
                else:
                    print(f"Skills: {skills}")
            
            # Print source
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"Error verifying storage: {str(e)}")
        return False

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Store document embeddings in Qdrant vector database")
    parser.add_argument("--embeddings-file", default="output/document_embeddings.json", 
                        help="Path to embeddings JSON file")
    parser.add_argument("--collection", default="document_embeddings", 
                        help="Name of the Qdrant collection")
    parser.add_argument("--batch-size", type=int, default=100, 
                        help="Number of embeddings to insert in a single batch")
    parser.add_argument("--verify", action="store_true", 
                        help="Verify storage with a test query")
    parser.add_argument("--query", default="data warehouse", 
                        help="Test query for verification")
    
    args = parser.parse_args()
    
    # Check if embeddings file exists
    if not os.path.exists(args.embeddings_file):
        print(f"Error: Embeddings file not found at {args.embeddings_file}")
        print("Run 'python run_docling.py --create-embeddings' first to create embeddings")
        return
        
    # Store embeddings in Qdrant
    result = store_embeddings_in_qdrant(
        embeddings_file=args.embeddings_file,
        collection_name=args.collection,
        batch_size=args.batch_size
    )
    
    # Verify storage if requested
    if args.verify and result.get("embeddings_inserted", 0) > 0:
        verify_qdrant_storage(
            collection_name=args.collection,
            query=args.query
        )

if __name__ == "__main__":
    main() 