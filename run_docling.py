from app.main import parse_document
from app.embedding_service import EmbeddingService
import os
import glob
import traceback
import json
import argparse
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process documents with Docling")
    parser.add_argument("--basic-chunking", action="store_true", help="Use basic chunking instead of advanced")
    parser.add_argument("--create-embeddings", action="store_true", help="Create embeddings for document chunks")
    args = parser.parse_args()
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    
    # Create configuration with skill extraction enabled using GPT-4o
    config = {
        "skill_extraction": {
            "enabled": True,
            "openai_api_key": openai_api_key,
            "method": "openai",
            "model": "gpt-4o",  # Explicitly use GPT-4o model
            "min_confidence": 0.5,
            "store_rejected": True, 
        },
        "structuring": {
            "use_advanced_chunking": True , # Enable advanced chunking by default
        }
    }
    
    # Load advanced chunking config if available
    advanced_config_path = os.path.join(current_dir, "examples", "advanced_chunking_config.json")
    if os.path.exists(advanced_config_path):
        try:
            with open(advanced_config_path, 'r') as f:
                advanced_config = json.load(f)
                if "structuring" in advanced_config:
                    config["structuring"].update(advanced_config["structuring"])
        except Exception as e:
            print(f"Warning: Failed to load advanced chunking config: {str(e)}")
    
    # Use basic chunking if specified
    if args.basic_chunking:
        config["structuring"]["use_advanced_chunking"] = False
        print("Basic chunking enabled (advanced chunking disabled)")
    else:
        print("Advanced chunking enabled (default)")
    
    # Enable embeddings if specified
    if args.create_embeddings:
        if "embeddings" not in config:
            config["embeddings"] = {}
        config["embeddings"]["enabled"] = True
        config["embeddings"]["openai_api_key"] = openai_api_key
        print("Embeddings creation enabled")
    
    # Get all image files from the directory
    img_dir = r"C:\Users\roaa.alashqar\Desktop\marketing-rag-docs"
    document_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                    glob.glob(os.path.join(img_dir, "*.png")) + \
                    glob.glob(os.path.join(img_dir, "*.pdf")) + \
                    glob.glob(os.path.join(img_dir, "*.webp"))
    
    if not document_paths:
        print(f"No image files found in {img_dir}")
        return
        
    print(f"Found {len(document_paths)} files to process")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {document_paths}")
    print(f"Skill extraction enabled with OpenAI GPT-4o model")
    if args.basic_chunking:
        print("Basic chunking enabled (advanced chunking disabled)")
    else:
        print("Advanced chunking enabled (default)")
    
    try:
        result = parse_document(document_paths, config=config, output_dir=output_dir)
        print("\nProcessing complete!")
        print(f"Processed {result['document_count']} document(s)")
        # Fix for KeyError: Check for both chunk_count and total_chunk_count
        if 'chunk_count' in result:
            print(f"Created {result['chunk_count']} chunks")
        elif 'total_chunk_count' in result:
            print(f"Created {result['total_chunk_count']} chunks")
        else:
            # Fallback if neither key exists
            print(f"Created chunks (count unavailable)")
        
        # Create embeddings directly if embeddings were requested but failed in the pipeline
        if args.create_embeddings and ("embedding_error" in result or "embeddings_path" not in result):
            print("\nRunning direct embeddings creation...")
            try:
                # Create embeddings using our EmbeddingService
                embedding_service = EmbeddingService()
                
                parsed_docs_path = result.get("output_path", os.path.join(output_dir, "parsed_documents.json"))
                embeddings_output_path = os.path.join(output_dir, "document_embeddings.json")
                
                # Process the JSON file to create embeddings
                embedding_result = embedding_service.process_json_file(parsed_docs_path)
                
                # Save embeddings to output file
                embedding_service.save_embeddings(embedding_result, embeddings_output_path)
                
                print(f"Direct embeddings creation complete")
                print(f"Created {len(embedding_result.get('processed_chunks', []))} embeddings")
                print(f"Embeddings saved to: {embeddings_output_path}")
            except Exception as e:
                print(f"Error creating embeddings directly: {str(e)}")
                traceback.print_exc()
        # Print embeddings information if available from the pipeline
        elif "embedding_count" in result:
            print(f"\nEmbeddings created: {result['embedding_count']}")
            print(f"Embeddings saved to: {result.get('embeddings_path')}")
            
        if "embedding_error" in result:
            print(f"\nWarning: Error creating embeddings: {result['embedding_error']}")
        
        # Check for consolidated skill data
        if "consolidated_skill_data" in result:
            skill_data = result["consolidated_skill_data"]
            validated_count = len(skill_data.get("validated_skills", []))
            rejected_count = len(skill_data.get("rejected_skills", []))
            
            print("\nConsolidated Skill Extraction Results:")
            print("===================================")
            print(f"Total validated skills: {validated_count}")
            print(f"Total rejected skills: {rejected_count}")
            
            # Print top skills if available
            if "top_skills" in skill_data and skill_data["top_skills"]:
                print("\nTop skills found across all documents:")
                for i, skill in enumerate(skill_data["top_skills"][:10], 1):  # Show top 10
                    # Fix for unhashable type error: Handle skills as dicts or strings
                    if isinstance(skill, dict):
                        skill_name = skill.get("name", "Unknown")
                        skill_count = skill.get("count", 0)
                        print(f"  {i}. {skill_name} (found in {skill_count} document{'s' if skill_count != 1 else ''})")
                    else:
                        # Original string-based skill handling
                        count = skill_data.get("skill_frequencies", {}).get(skill, 0)
                        print(f"  {i}. {skill} (found in {count} document{'s' if count != 1 else ''})")
            
            # Print all validated skills
            if validated_count > 0:
                print("\nAll validated skills:")
                for skill in skill_data.get("validated_skills", []):
                    # Fix for unhashable type error: Handle skills as dicts or strings
                    if isinstance(skill, dict):
                        skill_name = skill.get("name", "Unknown")
                        skill_count = skill.get("count", 0)
                        print(f"  - {skill_name} (found in {skill_count} document{'s' if skill_count != 1 else ''})")
                    else:
                        # Original string-based skill handling
                        count = skill_data.get("skill_frequencies", {}).get(skill, 0)
                        print(f"  - {skill} (found in {count} document{'s' if count != 1 else ''})")
            
            # Print rejected skills if available and requested
            if rejected_count > 0:
                print("\nRejected skills:")
                for skill in skill_data.get("rejected_skills", [])[:20]:  # Limit to first 20
                    print(f"  - {skill}")
                if len(skill_data.get("rejected_skills", [])) > 20:
                    print(f"  ... and {len(skill_data.get('rejected_skills', [])) - 20} more")
            
            # Print file paths to skill reports
            if "skill_state_path" in result:
                print(f"\nSkill state saved to: {result['skill_state_path']}")
            if "skill_report_path" in result:
                print(f"Detailed skill report saved to: {result['skill_report_path']}")
                
                # Print a tip about the report
                print("\nTip: The skill report contains a detailed breakdown of skills by document.")
                print("     Open it in a text editor to see the full analysis.")
        else:
            # If no consolidated skill data, print individual document skills
            skills_found = False
            for document in result.get('processed_documents', []):
                if document.get('metadata', {}).get('skill_extraction'):
                    skills_found = True
                    doc_id = document.get('metadata', {}).get('source', 'unknown')
                    print(f"\nExtracted Skills from {doc_id}:")
                    print("-----------------")
                    skills_data = document['metadata']['skill_extraction']
                    
                    if skills_data.get('validated_skills'):
                        print(f"Validated Skills: {', '.join(skills_data['validated_skills'])}")
                    
                    if skills_data.get('rejected_skills'):
                        print(f"Rejected Skills: {', '.join(skills_data['rejected_skills'])}")
            
            if not skills_found:
                print("\nNo skills were extracted from the documents")
            
        if "output_path" in result:
            print(f"\nResults saved to: {result['output_path']}")
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()