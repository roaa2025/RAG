import os
import argparse
from app.embedding_service import EmbeddingService
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create embeddings from parsed documents")
    parser.add_argument("--input", type=str, default="output/parsed_documents.json", 
                       help="Path to the input JSON file with parsed documents")
    parser.add_argument("--output", type=str, default="output/embeddings.json",
                       help="Path to save the embeddings output")
    args = parser.parse_args()
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    
    print(f"Creating embeddings from: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    try:
        # Initialize the embedding service
        print("Initializing embedding service...")
        embedding_service = EmbeddingService()
        
        # Process the JSON file
        print("Processing documents and creating embeddings...")
        result = embedding_service.process_json_file(args.input)
        
        # Print statistics
        chunk_count = len(result.get("processed_chunks", []))
        print(f"Created embeddings for {chunk_count} chunks")
        
        # Save the embeddings
        print("Saving embeddings to file...")
        output_path = embedding_service.save_embeddings(result, args.output)
        
        print(f"Embeddings successfully saved to: {output_path}")
        print(f"Total chunks processed: {chunk_count}")
        
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 