"""
Direct test of the skill extraction functionality without file processing.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path so we can import the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocessing.skill_extractor import SkillExtractor

def main():
    """
    Test the skill extractor directly using GPT-4o with fallback to mock.
    """
    # Set the OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY", "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")
    
    # Example text
    test_text = """
    John is proficient in Python, SQL, and has experience in machine learning. 
    He also enjoys working with data and coding. Additionally, he has skills in 
    natural language processing and deep learning frameworks like TensorFlow and PyTorch.
    He's interested in big data technologies such as Hadoop and Spark.
    """
    
    print("Processing text with skill extraction using GPT-4o model:")
    print("-" * 50)
    print(test_text)
    print("-" * 50)
    
    # First try with OpenAI GPT-4o model
    try:
        # Initialize the skill extractor with GPT-4o
        extractor_config_gpt4o = {
            "enabled": True,
            "openai_api_key": openai_api_key,
            "method": "openai",
            "model": "gpt-4o",
            "min_confidence": 0.5,
            "store_rejected": True
        }
        
        skill_extractor = SkillExtractor(extraction_config=extractor_config_gpt4o)
        skill_result = skill_extractor.extract_skills(test_text)
        
        print("\nSkill Extraction Results (using GPT-4o):")
        print("-" * 50)
        process_results(skill_result)
        
    except Exception as e:
        print(f"Error with GPT-4o API: {str(e)}")
        print("Falling back to mock extraction...")
        
        # Fallback to mock method
        extractor_config_mock = {
            "enabled": True,
            "method": "mock",
            "store_rejected": True
        }
        
        skill_extractor_mock = SkillExtractor(extraction_config=extractor_config_mock)
        skill_result = skill_extractor_mock.extract_skills(test_text)
        
        print("\nSkill Extraction Results (using mock method):")
        print("-" * 50)
        process_results(skill_result)

def process_results(skill_result):
    """Process and display the extraction results."""
    # Print the validated skills
    print("Validated Skills:")
    if skill_result["state"]["validated_skills"]:
        for skill in skill_result["state"]["validated_skills"]:
            print(f"  - {skill}")
    else:
        print("  No validated skills found")
    
    # Print the rejected skills
    print("\nRejected Skills:")
    if skill_result["state"]["rejected_skills"]:
        for skill in skill_result["state"]["rejected_skills"]:
            print(f"  - {skill}")
    else:
        print("  No rejected skills found")
    
    # Display the full JSON structure
    print("\nFull JSON Output:")
    print(json.dumps(skill_result, indent=2))

if __name__ == "__main__":
    main() 