import os
import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    # Import our document parser
    from app.main import parse_document
except ImportError as e:
    logging.error(f"Error importing required modules: {e}")
    logging.info("This script assumes you have the necessary packages installed")
    sys.exit(1)

def main():
    # Path to your documents
    docs_dir = Path(r"C:\Users\roaa.alashqar\Desktop\marketing-rag-docs")
    
    # Create output directory
    output_dir = Path("output") / "marketing_docs_chunking"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load advanced chunking configuration
    config_path = Path("examples") / "advanced_chunking_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            logging.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        config = {
            "structuring": {
                "use_advanced_chunking": True
            }
        }
        logging.info("Using default configuration with advanced chunking enabled")
    
    # Ensure advanced chunking is enabled
    if "structuring" not in config:
        config["structuring"] = {}
    config["structuring"]["use_advanced_chunking"] = True
    
    # Check if directory exists
    if not docs_dir.exists() or not docs_dir.is_dir():
        logging.error(f"Directory not found: {docs_dir}")
        print(f"\nThe directory {docs_dir} does not exist or is not accessible.")
        return
    
    # Find document files
    file_paths = []
    for ext in [".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".docx"]:
        file_paths.extend(list(docs_dir.glob(f"*{ext}")))
    
    if not file_paths:
        logging.error(f"No document files found in {docs_dir}")
        print(f"\nNo document files found in {docs_dir}.")
        print("Supported formats: PDF, PNG, JPG, JPEG, TXT, MD, DOCX")
        return
    
    # Log found files
    logging.info(f"Found {len(file_paths)} files in {docs_dir}")
    for file_path in file_paths:
        logging.info(f"File: {file_path.name}")
    
    print(f"\nProcessing {len(file_paths)} files with multi-layer chunking strategy...")
    
    # Convert Path objects to strings for processing
    str_file_paths = [str(path) for path in file_paths]
    
    try:
        # Process the documents
        result = parse_document(
            str_file_paths,
            config=config,
            output_dir=str(output_dir)
        )
        
        # Print summary of the results
        print("\nChunking complete!")
        print(f"Processed {result['document_count']} document(s)")
        
        if "total_chunk_count" in result:
            print(f"Generated {result['total_chunk_count']} total chunks")
        elif "chunk_count" in result:
            print(f"Generated {result['chunk_count']} chunks")
        
        # Print chunk evaluation if available
        if "chunk_evaluation" in result:
            for doc_id, eval_data in result["chunk_evaluation"].items():
                print(f"\nDocument: {doc_id}")
                print(f"  Total chunks: {eval_data.get('chunk_count', 0)}")
                
                # Print token statistics
                if "token_counts" in eval_data:
                    token_stats = eval_data["token_counts"]
                    print(f"  Token counts: min={token_stats.get('min', 0)}, "
                          f"max={token_stats.get('max', 0)}, "
                          f"avg={token_stats.get('avg', 0):.1f}")
                
                # Print semantic coherence if available
                if "semantic_coherence" in eval_data:
                    sem_stats = eval_data["semantic_coherence"]
                    print(f"  Semantic coherence: min={sem_stats.get('min', 0):.2f}, "
                          f"max={sem_stats.get('max', 0):.2f}, "
                          f"avg={sem_stats.get('avg', 0):.2f}")
                
                # Print chunk type distribution
                if "chunk_types" in eval_data:
                    print("  Chunk types:")
                    for chunk_type, count in eval_data["chunk_types"].items():
                        print(f"    {chunk_type}: {count}")
        
        # Print output path
        if "output_path" in result:
            print(f"\nResults saved to: {result['output_path']}")
            print("View the complete JSON output for detailed chunk information.")
    
    except Exception as e:
        logging.error(f"Error processing documents: {str(e)}")
        print(f"\nAn error occurred while processing the documents: {str(e)}")
        raise

if __name__ == "__main__":
    main() 