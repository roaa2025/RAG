"""
Example script demonstrating the skill extraction functionality.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path so we can import the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import parse_document

def main():
    """
    Example of using the document parser with skill extraction using GPT-4o.
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
    
    # Example text to test (using string input instead of a file)
    test_text = """
    John is proficient in Python, SQL, and has experience in machine learning. 
    He also enjoys working with data and coding. Additionally, he has skills in 
    natural language processing and deep learning frameworks like TensorFlow and PyTorch.
    He's interested in big data technologies such as Hadoop and Spark.
    """
    
    # Create a temporary text file to test
    temp_file_path = "temp_test_skills.txt"
    with open(temp_file_path, "w") as f:
        f.write(test_text)
    
    print("Processing text with skill extraction using GPT-4o model:")
    print("-" * 50)
    print(test_text)
    print("-" * 50)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the document with skill extraction
    try:
        result = parse_document(temp_file_path, config=config, output_dir=output_dir)
        
        print("\nSkill Extraction Results (using GPT-4o):")
        print("-" * 50)
        
        # Print the results
        if "processed_documents" in result:
            for doc in result["processed_documents"]:
                if "metadata" in doc and "skill_extraction" in doc["metadata"]:
                    skill_data = doc["metadata"]["skill_extraction"]
                    
                    print("Validated Skills:")
                    if skill_data.get("validated_skills"):
                        for skill in skill_data["validated_skills"]:
                            print(f"  - {skill}")
                    else:
                        print("  No validated skills found")
                    
                    print("\nRejected Skills:")
                    if skill_data.get("rejected_skills"):
                        for skill in skill_data["rejected_skills"]:
                            print(f"  - {skill}")
                    else:
                        print("  No rejected skills found")
                    
                    # Display the full JSON structure
                    print("\nFull JSON Output:")
                    print(json.dumps({
                        "text": test_text,
                        "state": {
                            "validated_skills": skill_data.get("validated_skills", []),
                            "rejected_skills": skill_data.get("rejected_skills", [])
                        }
                    }, indent=2))
    except Exception as e:
        print(f"Error during skill extraction: {str(e)}")
        print("Falling back to mock extraction method...")
        
        # Create new config with mock method
        mock_config = {
            "skill_extraction": {
                "enabled": True,
                "method": "mock",
                "store_rejected": True
            }
        }
        
        # Try again with mock method
        try:
            result = parse_document(temp_file_path, config=mock_config, output_dir=output_dir)
            print("\nSkill Extraction Results (using mock method):")
            print("-" * 50)
            
            for doc in result.get("processed_documents", []):
                if "metadata" in doc and "skill_extraction" in doc["metadata"]:
                    skill_data = doc["metadata"]["skill_extraction"]
                    
                    print("Validated Skills:")
                    if skill_data.get("validated_skills"):
                        for skill in skill_data["validated_skills"]:
                            print(f"  - {skill}")
                    else:
                        print("  No validated skills found")
                    
                    print("\nRejected Skills:")
                    if skill_data.get("rejected_skills"):
                        for skill in skill_data["rejected_skills"]:
                            print(f"  - {skill}")
                    else:
                        print("  No rejected skills found")
        except Exception as nested_e:
            print(f"Error with mock extraction as well: {str(nested_e)}")
    
    # Clean up the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

if __name__ == "__main__":
    main() 