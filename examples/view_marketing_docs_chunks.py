import json
import os
from pathlib import Path

def main():
    # Load the JSON results
    output_path = Path("output") / "marketing_docs_chunking" / "parsed_documents.json"
    
    if not output_path.exists():
        print(f"Result file not found at: {output_path}")
        return
    
    # Try multiple encodings to handle potential encoding issues
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    data = None
    
    for encoding in encodings:
        try:
            with open(output_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            print(f"Successfully loaded file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"Failed to load with {encoding} encoding, trying next...")
        except json.JSONDecodeError:
            print(f"Invalid JSON with {encoding} encoding, trying next...")
    
    if data is None:
        print("Failed to load the JSON file with any encoding. The file may be corrupted.")
        return
    
    # Print summary info
    print("===== Marketing Documents Chunking Results =====")
    print(f"Total documents: {data['document_count']}")
    print(f"Total chunks: {data['total_chunk_count']}")
    print(f"Input files:")
    for i, file_path in enumerate(data.get('input_files', [])):
        print(f"  {i+1}. {file_path}")
    
    # Print evaluation metrics
    if "chunk_evaluation" in data:
        print("\n===== Chunk Evaluation =====")
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
    
    # Print details about the chunks
    if "chunks" in data and len(data["chunks"]) > 0:
        print("\n===== Chunk Details =====")
        
        for i, chunk in enumerate(data["chunks"]):
            print(f"\nChunk {i+1}: {chunk.get('chunk_id', 'unknown')}")
            print(f"  Document: {chunk.get('document_id', 'unknown')}")
            print(f"  Type: {chunk.get('chunk_type', 'unknown')}")
            
            # Show hierarchical info if available
            if "section_header" in chunk:
                print(f"  Section: {chunk['section_header']} (Level {chunk.get('section_level', 0)})")
            
            # Show semantic info if available
            semantic_info = chunk.get("semantic_info", {})
            if semantic_info:
                print(f"  Semantic chunk: {semantic_info.get('index', 0)+1} of {semantic_info.get('total', 0)}")
            
            # Show fixed sizing info if available
            fixed_info = chunk.get("fixed_info", {})
            if fixed_info:
                print(f"  Fixed-size chunk: {fixed_info.get('index', 0)+1} of {fixed_info.get('total', 0)}")
            
            # Content info
            content = chunk.get("content", "")
            print(f"  Content length: {len(content)}")
            
            # Show first 200 chars of content, safely handling non-ASCII characters
            try:
                preview = content[:200].replace('\n', ' ')
                print(f"  Content preview: \"{preview}{'...' if len(content) > 200 else ''}\"")
            except UnicodeEncodeError:
                # Handle non-ASCII characters
                safe_preview = ''.join(c if ord(c) < 128 else '?' for c in content[:200]).replace('\n', ' ')
                print(f"  Content preview (ASCII only): \"{safe_preview}{'...' if len(content) > 200 else ''}\"")
    
    # Optionally save chunk content to separate files
    save_chunks = input("\nDo you want to save individual chunks to separate files for review? (y/n): ")
    if save_chunks.lower() == 'y':
        # Create directory for chunk files
        chunks_dir = Path("output") / "marketing_docs_chunking" / "chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        
        for i, chunk in enumerate(data.get("chunks", [])):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            doc_id = chunk.get("document_id", "unknown")
            chunk_type = chunk.get("chunk_type", "unknown")
            
            # Create filename
            filename = f"{doc_id}_{chunk_id}_{chunk_type}.txt"
            file_path = chunks_dir / filename
            
            try:
                # Write content to file with UTF-8 encoding
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Chunk ID: {chunk_id}\n")
                    f.write(f"Document ID: {doc_id}\n")
                    f.write(f"Chunk Type: {chunk_type}\n")
                    
                    if "section_header" in chunk:
                        f.write(f"Section: {chunk['section_header']} (Level {chunk.get('section_level', 0)})\n")
                    
                    # Add semantic info if available
                    if "semantic_info" in chunk:
                        f.write(f"Semantic chunk: {chunk['semantic_info'].get('index', 0)+1} of {chunk['semantic_info'].get('total', 0)}\n")
                    
                    # Add fixed size info if available
                    if "fixed_info" in chunk:
                        f.write(f"Fixed-size chunk: {chunk['fixed_info'].get('index', 0)+1} of {chunk['fixed_info'].get('total', 0)}\n")
                    
                    f.write("\n--- CONTENT ---\n\n")
                    f.write(chunk.get("content", ""))
            except UnicodeEncodeError:
                print(f"Warning: Encoding issues when saving {filename}. Some characters may not be preserved.")
                # Try with a different encoding
                with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(f"Chunk ID: {chunk_id}\n")
                    f.write(f"Document ID: {doc_id}\n")
                    f.write(f"Chunk Type: {chunk_type}\n")
                    f.write("Note: This file contains characters that couldn't be encoded properly.\n")
                    f.write("\n--- CONTENT ---\n\n")
                    f.write(chunk.get("content", ""))
        
        print(f"\nChunk files saved to: {chunks_dir}")
    
    print("\n===== End of Report =====")

if __name__ == "__main__":
    main() 