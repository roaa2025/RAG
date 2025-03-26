#!/usr/bin/env python
"""
Script to search for documents in Qdrant vector database by semantic similarity.
This allows you to find relevant documents by providing a natural language query.
The search uses vector embeddings to find semantically similar documents.
"""

import argparse
import json
from dotenv import load_dotenv
from app.embedding_service import EmbeddingService
from typing import List, Dict, Optional

def search_documents(
    query: str,
    collection_name: str = "document_embeddings",
    k: int = 5,
    filter_skill: Optional[str] = None,
    include_content: bool = True,
    show_skills: bool = True,
    output_file: Optional[str] = None
) -> List[Dict]:
    """
    Search for documents in Qdrant by semantic similarity.
    
    Args:
        query: The search query
        collection_name: Name of the Qdrant collection
        k: Number of results to return
        filter_skill: Optional skill to filter results by
        include_content: Whether to include document content in results
        show_skills: Whether to show extracted skills
        output_file: Optional file to save results to
        
    Returns:
        List of search results
    """
    try:
        # Initialize embedding service
        embedding_service = EmbeddingService()
        
        # Construct filter condition if skill is specified
        filter_condition = None
        if filter_skill:
            filter_condition = {
                "must": [
                    {"key": "metadata.skills", "match": {"value": filter_skill}}
                ]
            }
        
        # Perform search
        results = embedding_service.search_with_score(
            query=query,
            k=k,
            filter_condition=filter_condition
        )
        
        # Format results
        formatted_results = []
        for doc, score in results:
            result = {
                "similarity": score,
                "source": doc.metadata.get("source", "Unknown source")
            }
            
            if include_content:
                result["content"] = doc.page_content
                result["content_length"] = len(doc.page_content)  # Add content length
            
            if show_skills:
                result["skills"] = doc.metadata.get("skills", [])
                
            formatted_results.append(result)
            
        # Print results
        print(f"\nFound {len(results)} results for query: '{query}'\n")
        
        for i, result in enumerate(formatted_results, 1):
            print(f"Result {i} (Similarity: {result['similarity']:.3f}):")
            if include_content:
                print(f"Content: {result['content']}")
                print(f"Content length: {result['content_length']} characters")  # Print content length
            if show_skills and result.get("skills"):
                print("Skills:")
                for skill in result["skills"]:
                    print(f"- {skill}")
            print(f"Source: {result['source']}\n")
            
        # Save results if output file specified
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(formatted_results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")
            
        return formatted_results
        
    except Exception as e:
        print(f"Error searching documents: {str(e)}")
        return []

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search for documents in Qdrant vector database")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--collection", default="document_embeddings", 
                      help="Name of the Qdrant collection")
    parser.add_argument("--results", type=int, default=5, 
                      help="Number of results to return")
    parser.add_argument("--filter-skill", 
                      help="Only return documents with this skill")
    parser.add_argument("--no-content", action="store_true", 
                      help="Don't include document content in results")
    parser.add_argument("--no-skills", action="store_true", 
                      help="Don't show skills in results")
    parser.add_argument("--output", 
                      help="Save results to this file")
    
    args = parser.parse_args()
    
    # Search documents
    search_documents(
        query=args.query,
        collection_name=args.collection,
        k=args.results,
        filter_skill=args.filter_skill,
        include_content=not args.no_content,
        show_skills=not args.no_skills,
        output_file=args.output
    )

if __name__ == "__main__":
    main() 