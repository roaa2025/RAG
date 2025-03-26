#!/usr/bin/env python
"""
Test the multi-layer chunking strategy on example documents.
This script demonstrates the advanced chunking capabilities of the document parser.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path so we can import the app package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import parse_document
from app.utils.config import load_config_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def analyze_chunking_results(json_path):
    """Analyze the chunking results from the saved JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        print("\nChunking Analysis Results:")
        print(f"Processed {result['document_count']} document(s)")
        print(f"Generated {result.get('total_chunk_count', 0)} total chunks")
        
        # Print chunk evaluation summary
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
                
                # Print chunk size statistics
                if "chunk_sizes" in eval_data:
                    size_stats = eval_data["chunk_sizes"]
                    print(f"  Chunk sizes (chars): min={size_stats.get('min', 0)}, "
                          f"max={size_stats.get('max', 0)}, "
                          f"avg={size_stats.get('avg', 0):.1f}")
                
                # Print semantic coherence if available
                if "semantic_coherence" in eval_data:
                    sem_stats = eval_data["semantic_coherence"]
                    print(f"  Semantic coherence: min={sem_stats.get('min', 0):.2f}, "
                          f"max={sem_stats.get('max', 0):.2f}, "
                          f"avg={sem_stats.get('avg', 0):.2f}")
                
                # Print layer statistics
                if "layer_statistics" in eval_data:
                    print("\n  Layer Statistics:")
                    layer_stats = eval_data["layer_statistics"]
                    
                    # Hierarchical layer
                    if "hierarchical" in layer_stats:
                        hier_stats = layer_stats["hierarchical"]
                        print(f"    Hierarchical Layer:")
                        print(f"      Total chunks: {hier_stats['total_chunks']}")
                        print(f"      Unique sections: {hier_stats['unique_sections']}")
                    
                    # Semantic layer
                    if "semantic" in layer_stats:
                        sem_stats = layer_stats["semantic"]
                        print(f"    Semantic Layer:")
                        print(f"      Total chunks: {sem_stats['total_chunks']}")
                        print(f"      Unique groups: {sem_stats['unique_groups']}")
                    
                    # Fixed-size layer
                    if "fixed" in layer_stats:
                        fixed_stats = layer_stats["fixed"]
                        print(f"    Fixed-Size Layer:")
                        print(f"      Total chunks: {fixed_stats['total_chunks']}")
                        print(f"      Unique groups: {fixed_stats['unique_groups']}")
                
                # Print chunk type distribution
                if "chunk_types" in eval_data:
                    print("\n  Chunk Type Distribution:")
                    for chunk_type, count in eval_data["chunk_types"].items():
                        print(f"    {chunk_type}: {count}")
                
                # Print chunking pipeline distribution
                if "chunk_pipelines" in eval_data:
                    print("\n  Chunking Pipeline Distribution:")
                    for pipeline, count in eval_data["chunk_pipelines"].items():
                        print(f"    {pipeline}: {count}")
        
        # Print details about the chunks
        if "chunks" in result and len(result["chunks"]) > 0:
            print("\n===== Chunk Details =====")
            
            for i, chunk in enumerate(result["chunks"]):
                print(f"\nChunk {i+1}: {chunk.get('chunk_id', 'unknown')}")
                print(f"  Document: {chunk.get('document_id', 'unknown')}")
                print(f"  Type: {chunk.get('chunk_type', 'unknown')}")
                
                # Show layer information
                layer_info = chunk.get("layer_info", {})
                if layer_info:
                    print("\n  Layer Information:")
                    if "hierarchical" in layer_info:
                        hier_info = layer_info["hierarchical"]
                        print(f"    Hierarchical: Section '{hier_info.get('header', 'unknown')}' "
                              f"(Level {hier_info.get('level', 0)})")
                    if "semantic" in layer_info:
                        sem_info = layer_info["semantic"]
                        print(f"    Semantic: Group {sem_info.get('index', 0)+1} of {sem_info.get('total', 0)}")
                    if "fixed" in layer_info:
                        fixed_info = layer_info["fixed"]
                        print(f"    Fixed-Size: Group {fixed_info.get('index', 0)+1} of {fixed_info.get('total', 0)}")
                
                # Content info
                content = chunk.get("content", "")
                print(f"\n  Content length: {len(content)}")
                print(f"  Content preview: \"{content[:200]}{'...' if len(content) > 200 else ''}\"")
        
        print("\n===== End of Report =====")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing chunking results: {str(e)}")
        raise

def main():
    """Run the multi-layer chunking test on example documents"""
    # Determine the examples directory path
    examples_dir = Path(__file__).parent
    config_path = examples_dir / "advanced_chunking_config.json"
    output_dir = Path.cwd() / "output" / "multi_layer_chunking_test"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for example documents
    example_docs_dir = examples_dir / "documents"
    
    if not example_docs_dir.exists():
        logger.error(f"Example documents directory not found: {example_docs_dir}")
        print("\nPlease place example documents in the examples/documents directory.")
        print("You can include PDFs, images, or text files to test the chunking strategy.")
        
        # Create example docs directory if it doesn't exist
        os.makedirs(example_docs_dir, exist_ok=True)
        return
    
    # Find example documents
    example_files = []
    for ext in [".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"]:
        example_files.extend(list(example_docs_dir.glob(f"*{ext}")))
    
    if not example_files:
        logger.error("No example documents found")
        print("\nPlease add some document files (PDF, images, text) to examples/documents/ directory.")
        return
    
    # Load configuration
    if config_path.exists():
        config = load_config_from_file(str(config_path))
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Configuration file not found: {config_path}")
        config = {
            "structuring": {
                "use_advanced_chunking": True
            }
        }
    
    # Ensure advanced chunking is enabled
    if "structuring" not in config:
        config["structuring"] = {}
    config["structuring"]["use_advanced_chunking"] = True
    
    # Process the documents
    print(f"\nProcessing {len(example_files)} documents with multi-layer chunking strategy")
    
    # Convert Path objects to strings
    file_paths = [str(path) for path in example_files]
    
    try:
        # Parse the documents
        result = parse_document(
            file_paths,
            config=config,
            output_dir=str(output_dir)
        )
        
        # Analyze the results from the saved JSON file
        if "output_path" in result:
            json_path = result["output_path"]
            print(f"\nAnalyzing results from: {json_path}")
            analyze_chunking_results(json_path)
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

if __name__ == "__main__":
    main() 