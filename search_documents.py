#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command-line tool for performing semantic searches on documents stored in Qdrant.

This script provides a simple interface to search documents stored in the Qdrant
vector database using LangChain's similarity search capabilities.

Usage:
    python search_documents.py "your search query here" [options]
    
Example:
    python search_documents.py "data processing techniques" --top_k 5 --with_scores
"""

import sys
import os
from app.qdrant_search import QdrantSearchService

def main():
    # Check if a search query was provided
    if len(sys.argv) < 2:
        print("Error: You must provide a search query.")
        print("Usage: python search_documents.py \"your search query here\" [--top_k N] [--with_scores]")
        return 1
    
    # Extract the search query (first argument)
    query = sys.argv[1]
    
    # Parse optional arguments
    top_k = 3
    with_scores = False
    host = "localhost"
    port = 6333
    collection = "document_embeddings"
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # Handle --param=value format
        if "=" in arg and arg.startswith("--"):
            param, value = arg.split("=", 1)
            param = param.strip()
            value = value.strip()
            
            if param == "--top_k":
                try:
                    top_k = int(value)
                except ValueError:
                    print(f"Error: --top_k requires an integer value, got '{value}'")
                    return 1
            elif param == "--host":
                host = value.strip('"\'')  # Remove quotes if present
            elif param == "--port":
                try:
                    port = int(value)
                except ValueError:
                    print(f"Error: --port requires an integer value, got '{value}'")
                    return 1
            elif param == "--collection":
                collection = value
            else:
                print(f"Warning: Ignoring unknown parameter: {param}")
            i += 1
        # Handle --param value format
        elif arg.startswith("--"):
            param = arg.strip()
            
            if param == "--with_scores":
                with_scores = True
                i += 1
            elif param == "--top_k" and i + 1 < len(sys.argv):
                try:
                    top_k = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: --top_k requires an integer value, got '{sys.argv[i + 1]}'")
                    return 1
            elif param == "--host" and i + 1 < len(sys.argv):
                host = sys.argv[i + 1].strip('"\'')  # Remove quotes if present
                i += 2
            elif param == "--port" and i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: --port requires an integer value, got '{sys.argv[i + 1]}'")
                    return 1
            elif param == "--collection" and i + 1 < len(sys.argv):
                collection = sys.argv[i + 1]
                i += 2
            else:
                print(f"Warning: Ignoring unknown argument: {arg}")
                i += 1
        else:
            print(f"Warning: Ignoring unknown argument: {arg}")
            i += 1
            
    # Special handling for in-memory mode
    if host.lower() == ":memory:":
        print("Using in-memory Qdrant mode")
        host = ":memory:"
    
    # Initialize search service
    try:
        print(f"Initializing search service with host='{host}', collection='{collection}'")
        search_service = QdrantSearchService(
            collection_name=collection,
            host=host,
            port=port
        )
        
        print("\n================================")
        print("💫 Document Semantic Search 💫")
        print("================================")
        
        # Get collection stats
        print("Retrieving collection stats...")
        stats = search_service.get_collection_stats()
        print(f"Stats received: {stats}")
        
        if "error" in stats:
            print(f"\n❌ Error connecting to Qdrant: {stats['error']}")
            
            if host != ":memory:":
                print("\n💡 Tip: No Qdrant server detected. Try using in-memory mode:")
                print("   python setup_memory_qdrant.py")
                print("   python search_documents.py \"your query\" --host=\":memory:\"")
            else:
                print("\n❌ Error with in-memory mode. Try running setup_memory_qdrant.py first.")
            return 1
        
        print("\n📊 Collection Stats:")
        print(f"   Collection: {stats.get('collection_name', 'N/A')}")
        print(f"   Documents: {stats.get('vectors_count', 'N/A')}")
        print(f"   Vector Size: {stats.get('vector_size', 'N/A')}")
        print(f"   Distance Metric: {stats.get('distance', 'N/A')}")
        
        if stats.get('vectors_count', 0) == 0:
            print("\n⚠️ Warning: Collection exists but contains no vectors.")
            if host == ":memory:":
                print("   Try running 'python setup_memory_qdrant.py' first to create sample data.")
            return 1
        
        print(f"\n🔍 Searching for: \"{query}\"")
        print(f"   Returning top {top_k} results")
        
        # Perform search
        if with_scores:
            results = search_service.similarity_search_with_scores(query, k=top_k)
            
            if not results:
                print("\n❌ No results found.")
                return 0
                
            print(f"\n✅ Found {len(results)} results:\n")
            
            for i, (doc, score) in enumerate(results):
                print(f"Result #{i+1} [Similarity: {score:.4f}]")
                print("=" * 50)
                print(f"📄 Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)
                print("\n🏷️  Metadata:")
                
                for meta_key, meta_value in doc.metadata.items():
                    meta_str = str(meta_value)
                    if len(meta_str) > 100:
                        print(f"  • {meta_key}: {meta_str[:100]}...")
                    else:
                        print(f"  • {meta_key}: {meta_str}")
                print()
        else:
            results = search_service.similarity_search(query, k=top_k)
            
            if not results:
                print("\n❌ No results found.")
                return 0
                
            print(f"\n✅ Found {len(results)} results:\n")
            
            for i, doc in enumerate(results):
                print(f"Result #{i+1}")
                print("=" * 50)
                print(f"📄 Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)
                print("\n🏷️  Metadata:")
                
                for meta_key, meta_value in doc.metadata.items():
                    meta_str = str(meta_value)
                    if len(meta_str) > 100:
                        print(f"  • {meta_key}: {meta_str[:100]}...")
                    else:
                        print(f"  • {meta_key}: {meta_str}")
                print()
    
    except Exception as e:
        print(f"\n❌ Error performing search: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 