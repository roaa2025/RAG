"""
Example script demonstrating consolidated skill extraction from multiple documents.
This script shows how to extract skills from multiple documents and 
consolidate them into a unified skill state.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path so we can import the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import parse_document
from app.utils.skill_utils import process_skill_data

def main():
    """
    Example of extracting and consolidating skills from multiple documents.
    """
    # Set the OpenAI API key (replace with your own or use environment variable)
    openai_api_key = os.environ.get("OPENAI_API_KEY", "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")
    
    # Define a configuration with skill extraction enabled using GPT-4o
    config = {
        "skill_extraction": {
            "enabled": True,
            "openai_api_key": openai_api_key,
            "method": "openai",
            "model": "gpt-4o",  # Explicitly use the GPT-4o model
            "min_confidence": 0.5,
            "store_rejected": True
        }
    }
    
    # Example texts for multiple documents
    example_texts = [
        # Document 1: Software Developer
        """
        John is proficient in Python, SQL, and has experience in machine learning. 
        He also enjoys working with data and coding. Additionally, he has skills in 
        natural language processing and deep learning frameworks like TensorFlow and PyTorch.
        He's interested in big data technologies such as Hadoop and Spark.
        """,
        
        # Document 2: Data Scientist
        """
        Maria is an experienced data scientist with expertise in statistical analysis,
        R programming, and Python. She specializes in creating machine learning models
        and has worked extensively with scikit-learn, TensorFlow, and Keras.
        She also has experience with data visualization tools like Tableau and PowerBI.
        """,
        
        # Document 3: DevOps Engineer
        """
        Ahmed is a DevOps engineer with skills in Docker, Kubernetes, and AWS.
        He has experience with CI/CD pipelines using Jenkins and GitHub Actions.
        He's proficient in shell scripting, Python, and has worked with infrastructure
        as code tools like Terraform and Ansible.
        """
    ]
    
    # Create temporary text files
    temp_files = []
    for i, text in enumerate(example_texts):
        temp_file_path = f"temp_document_{i+1}.txt"
        with open(temp_file_path, "w") as f:
            f.write(text)
        temp_files.append(temp_file_path)
    
    print(f"Created {len(temp_files)} temporary documents for processing")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/multi_doc_skills")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\nStep 1: Processing documents with skill extraction")
        print("=" * 50)
        
        # Process the documents and extract skills
        result = parse_document(temp_files, config=config, output_dir=output_dir)
        
        print(f"\nProcessed {result['document_count']} documents")
        print(f"Generated {result['chunk_count']} chunks")
        
        # Print consolidated skill results
        if "consolidated_skill_data" in result:
            print("\nStep 2: Analyzing consolidated skill data")
            print("=" * 50)
            
            skill_data = result["consolidated_skill_data"]
            validated_count = len(skill_data.get("validated_skills", []))
            rejected_count = len(skill_data.get("rejected_skills", []))
            
            print(f"Total validated skills: {validated_count}")
            print(f"Total rejected skills: {rejected_count}")
            
            # Print top skills
            if "top_skills" in skill_data and skill_data["top_skills"]:
                print("\nTop skills found across all documents:")
                for i, skill in enumerate(skill_data["top_skills"], 1):
                    count = skill_data.get("skill_frequencies", {}).get(skill, 0)
                    print(f"  {i}. {skill} (found in {count} document{'s' if count != 1 else ''})")
            
            # Print skills by document
            print("\nSkills by document:")
            for doc_id, skills in skill_data.get("document_skill_mapping", {}).items():
                print(f"\nDocument: {doc_id}")
                print(f"  Validated skills: {', '.join(skills.get('validated', []))}")
            
            # Print file paths for generated reports
            if "skill_state_path" in result:
                print(f"\nSkill state saved to: {result['skill_state_path']}")
            if "skill_report_path" in result:
                print(f"Skill report saved to: {result['skill_report_path']}")
                
                # Print the report contents
                print("\nContents of the skill report:")
                print("-" * 50)
                try:
                    with open(result["skill_report_path"], "r") as f:
                        report = f.read()
                        print(report)
                except Exception as e:
                    print(f"Error reading report: {str(e)}")
        else:
            print("\nNo consolidated skill data found in the result.")
            
            # Manual consolidation as fallback
            print("\nStep 2 (Alternative): Manually consolidating skills")
            print("=" * 50)
            
            if "processed_documents" in result:
                print("Manually consolidating skills from processed documents...")
                skill_data = process_skill_data(
                    processed_documents=result["processed_documents"],
                    output_dir=output_dir,
                    include_rejected=True
                )
                
                # Print the consolidated skill state
                if "skill_state" in skill_data:
                    state = skill_data["skill_state"]
                    print(f"\nManually consolidated {len(state.get('validated_skills', []))} validated skills")
                    print(f"Top skills: {', '.join(state.get('top_skills', []))}")
                    
                    # Print file paths
                    if "state_path" in skill_data:
                        print(f"\nManually generated skill state saved to: {skill_data['state_path']}")
                    if "report_path" in skill_data:
                        print(f"Manually generated skill report saved to: {skill_data['report_path']}")
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print("\nCleaned up temporary files")

if __name__ == "__main__":
    main() 