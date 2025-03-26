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
    print("===== Multi-Layer Chunking Results Summary =====")
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
    
    # Track the multi-layer chunking process
    chunk_hierarchy = {}
    
    # Print details about the chunks
    if "chunks" in data and len(data["chunks"]) > 0:
        print("\n===== Multi-Layer Chunk Details =====")
        
        for i, chunk in enumerate(data["chunks"]):
            print(f"\nChunk {i+1}: {chunk.get('chunk_id', 'unknown')}")
            print(f"  Type: {chunk.get('chunk_type', 'unknown')}")
            
            # Show hierarchical info if available
            if "section_header" in chunk:
                print(f"  Section: {chunk['section_header']} (Level {chunk.get('section_level', 0)})")
                
                # Track hierarchy
                section = chunk['section_header']
                if section not in chunk_hierarchy:
                    chunk_hierarchy[section] = {"semantic": {}, "fixed": []}
            
            # Show semantic info if available
            semantic_info = chunk.get("semantic_info", {})
            if semantic_info:
                print(f"  Semantic chunk: {semantic_info.get('index', 0)+1} of {semantic_info.get('total', 0)}")
                
                # Track semantic grouping
                if "section_header" in chunk:
                    section = chunk['section_header']
                    sem_idx = semantic_info.get('index', 0)
                    if sem_idx not in chunk_hierarchy[section]["semantic"]:
                        chunk_hierarchy[section]["semantic"][sem_idx] = []
                    chunk_hierarchy[section]["semantic"][sem_idx].append(i)
            
            # Show fixed sizing info if available
            fixed_info = chunk.get("fixed_info", {})
            if fixed_info:
                print(f"  Fixed-size chunk: {fixed_info.get('index', 0)+1} of {fixed_info.get('total', 0)}")
                
                # Track fixed chunks
                if "section_header" in chunk:
                    section = chunk['section_header']
                    chunk_hierarchy[section]["fixed"].append(i)
            
            # Content info
            print(f"  Content length: {len(chunk.get('content', ''))}")
            # Show first 100 chars of content
            content = chunk.get("content", "")
            print(f"  Content preview: \"{content[:100]}{'...' if len(content) > 100 else ''}\"")
    
    # Print chunk hierarchy summary
    print("\n===== Chunking Hierarchy Summary =====")
    for section, info in chunk_hierarchy.items():
        print(f"\nSection: {section}")
        
        if info["semantic"]:
            print(f"  Semantic chunks: {len(info['semantic'])}")
            for sem_idx, chunk_indices in info["semantic"].items():
                print(f"    Group {sem_idx+1}: {len(chunk_indices)} fixed-size chunks")
        else:
            print(f"  Fixed-size chunks: {len(info['fixed'])}")
    
    print("\n===== End of Report =====")

if __name__ == "__main__":
    main() 