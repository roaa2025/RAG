import json
import os
from pathlib import Path

def main():
    # Load the JSON results
    output_path = Path("output") / "multi_layer_chunking_test" / "test_document_processed.json"
    
    if not output_path.exists():
        print(f"Result file not found at: {output_path}")
        return
    
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    # Print summary info
    print("===== Chunking Results Summary =====")
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
    
    # Print first 5 chunks info
    if "chunks" in data and len(data["chunks"]) > 0:
        print("\n===== First Chunks Details =====")
        max_chunks = min(5, len(data["chunks"]))
        
        for i in range(max_chunks):
            chunk = data["chunks"][i]
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk.get('chunk_id', 'unknown')}")
            print(f"  Type: {chunk.get('chunk_type', 'unknown')}")
            print(f"  Content length: {len(chunk.get('content', ''))}")
            print(f"  Content type: {chunk.get('content_type', 'unknown')}")
            
            # Show section header if available
            if "section_header" in chunk:
                print(f"  Section: {chunk['section_header']} (Level {chunk.get('section_level', 0)})")
            
            # Show semantic info if available
            if "semantic_info" in chunk:
                sem_info = chunk["semantic_info"]
                print(f"  Semantic info: Index {sem_info.get('index', 0)} of {sem_info.get('total', 0)}")
            
            # Show fixed sizing info if available
            if "fixed_info" in chunk:
                fixed_info = chunk["fixed_info"]
                print(f"  Fixed-size info: Index {fixed_info.get('index', 0)} of {fixed_info.get('total', 0)}")
            
            # Show first 200 chars of content
            content = chunk.get("content", "")
            print(f"\n  Content preview: \"{content[:200]}{'...' if len(content) > 200 else ''}\"")
    
    print("\n===== End of Report =====")

if __name__ == "__main__":
    main() 