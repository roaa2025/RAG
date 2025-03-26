#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides functionality to query the Qdrant vector database
using LangChain's similarity search capabilities.

It connects to a Qdrant collection containing document embeddings
and provides methods to search for semantically similar documents.
"""

from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

class QdrantSearchService:
    def __init__(
        self,
        collection_name: str = "document_embeddings",
        host: str = "localhost",
        port: int = 6333,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the Qdrant search service.
        
        Args:
            collection_name: Name of the Qdrant collection to search
            host: Qdrant server host (use ":memory:" for in-memory mode)
            port: Qdrant server port
            embedding_model: Name of the OpenAI embedding model to use
        """
        # Initialize the embedding model
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Qdrant client
        if host == ":memory:":
            print("Using in-memory Qdrant instance")
            self.qdrant_client = QdrantClient(":memory:")
        else:
            print(f"Connecting to Qdrant at {host}:{port}")
            self.qdrant_client = QdrantClient(host=host, port=port)
        
        # Initialize LangChain's Qdrant vectorstore
        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )
        
        self.collection_name = collection_name
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 3,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform a similarity search on the Qdrant collection.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_condition: Optional filter condition for Qdrant
            
        Returns:
            List of Documents with content and metadata
        """
        # Create embedding for the query
        print(f"Searching for: '{query}'")
        
        # Perform similarity search
        if filter_condition:
            documents = self.vectorstore.similarity_search(
                query=query, 
                k=k,
                filter=filter_condition
            )
        else:
            documents = self.vectorstore.similarity_search(
                query=query, 
                k=k
            )
        
        return documents
    
    def similarity_search_with_scores(
        self, 
        query: str, 
        k: int = 3,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Perform a similarity search on the Qdrant collection and return scores.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_condition: Optional filter condition for Qdrant
            
        Returns:
            List of tuples with (Document, score)
        """
        print(f"Searching with scores for: '{query}'")
        
        # Perform similarity search with scores
        if filter_condition:
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query=query, 
                k=k,
                filter=filter_condition
            )
        else:
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query=query, 
                k=k
            )
        
        return docs_and_scores
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Qdrant collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            print(f"Getting collection info for '{self.collection_name}'...")
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Got collection info, getting count...")
            vectors_count = self.qdrant_client.count(self.collection_name).count
            print(f"Count: {vectors_count}")
            
            stats = {
                "collection_name": self.collection_name,
                "vectors_count": vectors_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": str(collection_info.config.params.vectors.distance),
            }
            
            return stats
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Search for documents in Qdrant using similarity search"
    )
    parser.add_argument(
        "query", 
        type=str,
        help="Search query"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="document_embeddings",
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=3,
        help="Number of results to return"
    )
    parser.add_argument(
        "--with_scores", 
        action="store_true",
        help="Include similarity scores in the output"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="Qdrant host"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6333,
        help="Qdrant port"
    )
    
    args = parser.parse_args()
    
    # Initialize search service
    search_service = QdrantSearchService(
        collection_name=args.collection,
        host=args.host,
        port=args.port
    )
    
    # Get collection stats
    stats = search_service.get_collection_stats()
    print("\n=== Collection Stats ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
    
    # Perform search
    if args.with_scores:
        results = search_service.similarity_search_with_scores(args.query, k=args.top_k)
        print(f"\nTop {len(results)} results with scores:")
        for i, (doc, score) in enumerate(results):
            print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
            print(f"Content: {doc.page_content[:150]}..." if len(doc.page_content) > 150 else doc.page_content)
            print("Metadata:")
            for meta_key, meta_value in doc.metadata.items():
                meta_str = str(meta_value)
                print(f"  {meta_key}: {meta_str[:100]}..." if len(meta_str) > 100 else meta_str)
    else:
        results = search_service.similarity_search(args.query, k=args.top_k)
        print(f"\nTop {len(results)} results:")
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Content: {doc.page_content[:150]}..." if len(doc.page_content) > 150 else doc.page_content)
            print("Metadata:")
            for meta_key, meta_value in doc.metadata.items():
                meta_str = str(meta_value)
                print(f"  {meta_key}: {meta_str[:100]}..." if len(meta_str) > 100 else meta_str) 