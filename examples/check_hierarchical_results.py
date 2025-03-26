import json
import os
from pathlib import Path

def main():
    # Load the JSON results
    output_path = Path("output") / "hierarchical_chunking_test" / "test_document_processed.json"
    
    if not output_path.exists():
        print(f"Result file not found at: {output_path}")
        return
    
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    # Print summary info
    print("===== Hierarchical Chunking Results Summary =====")
    print(f"Total documents: {data['document_count']}")
    print(f"Total chunks: {data['total_chunk_count']}")
    
    # Print evaluation metrics
    if "chunk_evaluation" in data:
        for doc_id, eval_data in data["chunk_evaluation"].items():
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
            
            # Print chunk type distribution
            if "chunk_types" in eval_data:
                print("  Chunk types:")
                for chunk_type, count in eval_data["chunk_types"].items():
                    print(f"    {chunk_type}: {count}")
    
    # Print all chunks with their headers
    if "chunks" in data and len(data["chunks"]) > 0:
        print("\n===== All Hierarchical Chunks =====")
        
        for i, chunk in enumerate(data["chunks"]):
            print(f"\nChunk {i+1}: {chunk.get('section_header', 'No Header')} (Level {chunk.get('section_level', 0)})")
            print(f"  ID: {chunk.get('chunk_id', 'unknown')}")
            print(f"  Type: {chunk.get('chunk_type', 'unknown')}")
            print(f"  Content length: {len(chunk.get('content', ''))}")
            
            # Show content preview
            content = chunk.get("content", "")
            print(f"  Content preview: \"{content[:100]}{'...' if len(content) > 100 else ''}\"")
    
    print("\n===== End of Report =====")

if __name__ == "__main__":
    main() 