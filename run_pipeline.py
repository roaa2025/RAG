#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the complete document processing pipeline:
1. Document extraction (OCR if needed)
2. Text preprocessing
3. Skill extraction
4. Advanced chunking
5. Embedding generation
6. Vector store integration (Qdrant)

Example usage:
    python run_pipeline.py test_files/resume.pdf --output-dir output
    
Features:
- Processes one or more documents through the entire pipeline
- Uses advanced chunking strategies for optimal text segmentation
- Extracts skills and metadata from documents
- Creates embeddings for document chunks
- Stores results in Qdrant vector store for retrieval
- Saves results at each stage for analysis
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from app.main import run_complete_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Run the complete pipeline with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the complete document processing pipeline")
    parser.add_argument("file_paths", nargs="+", help="Path(s) to document(s) to process")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--no-qdrant-store", action="store_true", help="Skip storing in Qdrant")
    parser.add_argument("--collection", default="document_embeddings", help="Qdrant collection name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure all file paths exist
    valid_paths = []
    for path in args.file_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            logger.warning(f"File not found: {path}")
    
    if not valid_paths:
        logger.error("No valid files to process")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up config path if provided
    config_path = None
    if args.config:
        if os.path.exists(args.config):
            config_path = args.config
        else:
            logger.warning(f"Config file not found: {args.config}")
    
    # Run the pipeline
    print(f"\n🚀 Starting document processing pipeline for {len(valid_paths)} document(s):")
    for path in valid_paths:
        print(f"  - {path}")
    print(f"💾 Output directory: {args.output_dir}")
    
    try:
        # Run the complete pipeline
        results = run_complete_pipeline(
            file_paths=valid_paths,
            output_dir=args.output_dir,
            config_path=config_path,
            store_in_qdrant=not args.no_qdrant_store,
            qdrant_collection=args.collection
        )
        
        # Print summary
        print("\n✅ Pipeline completed successfully!")
        print(f"📄 Processed {len(valid_paths)} document(s)")
        
        # Print chunk information
        if "total_chunk_count" in results:
            print(f"🧩 Generated {results['total_chunk_count']} chunks")
        elif "chunk_count" in results:
            print(f"🧩 Generated {results['chunk_count']} chunks")
        
        # Print skill information
        if "consolidated_skill_data" in results:
            skill_data = results["consolidated_skill_data"]
            validated_count = len(skill_data.get("validated_skills", []))
            print(f"🔍 Extracted {validated_count} validated skills")
        
        # Print embedding information
        if "embedding_count" in results:
            print(f"🔢 Created {results['embedding_count']} embeddings")
        
        # Print Qdrant storage information
        if "qdrant_storage" in results:
            qdrant_info = results["qdrant_storage"]
            print(f"💾 Stored {qdrant_info.get('embeddings_stored', 0)} embeddings in Qdrant")
            print(f"⏱️ Vector storage completed in {qdrant_info.get('elapsed_seconds', 0):.2f} seconds")
        
        # Print output file locations
        print("\n📂 Result files:")
        if "output_path" in results:
            print(f"  - Parsed documents: {results['output_path']}")
        if "embeddings_path" in results:
            print(f"  - Embeddings: {results['embeddings_path']}")
        if "skill_state_path" in results:
            print(f"  - Skill state: {results['skill_state_path']}")
        if "skill_report_path" in results:
            print(f"  - Skill report: {results['skill_report_path']}")
        
        # Print search instructions
        print("\n🔍 To search the stored documents:")
        print("1. Run a search using memory_search.py or search_documents.py")
        print("2. Example: python memory_search.py --query \"your query here\"")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 