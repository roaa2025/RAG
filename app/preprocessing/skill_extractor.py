from typing import Dict, List, Any, Optional, Set, Union
import logging
import re
import os
import json
from openai import OpenAI  # Import new OpenAI client
from langchain.schema import Document

class SkillExtractor:
    """
    Class for extracting, validating, and categorizing domain-specific skills from text.
    Uses NER (Named Entity Recognition) to identify skill candidates and validates them
    with LLM-based contextual analysis.
    """
    
    def __init__(self, extraction_config: Optional[Dict] = None):
        """
        Initialize the skill extractor with optional configuration.
        
        Args:
            extraction_config: Optional configuration for skill extraction
        """
        self.logger = logging.getLogger(__name__)
        
        # Set default configuration
        self.extraction_config = extraction_config or {
            "enabled": True,
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "method": "openai",
            "model": "gpt-4o",  # Use gpt-4o as the default model
            "min_confidence": 0.5,
            "store_rejected": True,
        }
        
        # Set OpenAI API key and initialize client
        openai_api_key = self.extraction_config.get("openai_api_key", "")
        
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = OpenAI()
        
        self.logger.info("Initialized SkillExtractor with model: %s", self.extraction_config.get("model"))
        
    def _extract_skills_with_openai(self, text: str) -> Dict[str, List[str]]:
        """
        Extract potential skills from text using OpenAI's NER capabilities.
        
        Args:
            text: Text to extract skills from
            
        Returns:
            Dictionary with extracted skills
        """
        if not text or not self.extraction_config.get("openai_api_key"):
            return {"extracted_skills": []}

        try:
            # Get the model from config
            model = self.extraction_config.get("model", "gpt-4o")
            
            # Use chat completion API with new client
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": 
                        "You are an AI specialized in Named Entity Recognition (NER). "
                        "Your task is to extract all potential skills, technologies, and competencies from text. "
                        "Consider variations and abbreviations (e.g., 'ML' = 'Machine Learning'). "
                        "Identify only clear skills, not vague terms. Return only the extracted skills as a list."
                    },
                    {"role": "user", "content": f"Extract skills from this text: {text}"}
                ],
                temperature=0.3,
                max_tokens=100
            )
            skill_list_text = response.choices[0].message.content.strip()
            
            # Clean up the response to extract skills
            # Handle lists like "1. Python 2. SQL" or "- Python, - SQL" or just "Python, SQL"
            skills = []
            
            # Remove list markers and split by newlines or commas
            cleaned_text = re.sub(r'^[-*•]|\d+\.\s+', '', skill_list_text, flags=re.MULTILINE)
            
            # Split by newlines and commas, clean each skill
            for line in cleaned_text.split('\n'):
                for skill in line.split(','):
                    skill = skill.strip()
                    if skill and len(skill) > 1:  # Avoid single-character skills or empty strings
                        skills.append(skill)
            
            self.logger.info(f"Extracted {len(skills)} skills from text using model: {model}")
            return {"extracted_skills": skills}
            
        except Exception as e:
            self.logger.error(f"Error extracting skills with OpenAI: {str(e)}")
            return {"extracted_skills": []}
    
    def _validate_skills_with_openai(self, text: str, extracted_skills: List[str]) -> Dict[str, Any]:
        """
        Validate and categorize extracted skills using OpenAI's contextual analysis.
        Also adds confidence scores to each validated skill.
        
        Args:
            text: Original text
            extracted_skills: List of initially extracted skills
            
        Returns:
            Dictionary with validated and rejected skills, including confidence scores
        """
        if not text or not extracted_skills or not self.extraction_config.get("openai_api_key"):
            return {"validated_skills": [], "rejected_skills": []}

        try:
            # Prepare the extracted skills as a comma-separated string
            skills_string = ", ".join(extracted_skills)
            
            # Get the model from config
            model = self.extraction_config.get("model", "gpt-4o")
            
            # Use chat completion API with new client
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": 
                        "You are an AI specialized in contextual analysis for skill verification. "
                        "Your task is to validate whether each extracted skill is relevant and meaningful "
                        "based on the surrounding text. Normalize skills to standard names. "
                        "For each validated skill, provide a confidence score between 0.0 and 1.0, where: "
                        "1.0 = extremely confident this is a genuine skill mentioned in the text, "
                        "0.5 = moderately confident, "
                        "0.0 = not confident at all. "
                        "Categorize skills into 'validated_skills' (contextually relevant) with their confidence scores, "
                        "and 'rejected_skills' (generic, out of context) lists. "
                        "Return the results in JSON format with 'validated_skills' (array of objects with 'name' and 'confidence' fields) "
                        "and 'rejected_skills' (array of strings)."
                    },
                    {"role": "user", "content": 
                        f"Original text: {text}\n\nExtracted skills to validate: {skills_string}\n\n"
                        "Validate these skills and return a JSON with validated_skills (array of objects with name and confidence) and rejected_skills arrays."
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            response_text = response.choices[0].message.content.strip()
            
            # Extract the JSON part from the response
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                    
                    # Ensure validated_skills has the proper format (name and confidence)
                    validated_skills = result.get("validated_skills", [])
                    structured_validated_skills = []
                    
                    # Apply minimum confidence threshold
                    min_confidence = self.extraction_config.get("min_confidence", 0.5)
                    
                    # Process validated skills
                    for skill in validated_skills:
                        # Handle both object format and legacy string format
                        if isinstance(skill, dict):
                            # Modern format
                            if skill.get("confidence", 0) >= min_confidence:
                                structured_validated_skills.append(skill)
                        else:
                            # Legacy format - convert to object with default confidence
                            structured_validated_skills.append({
                                "name": skill,
                                "confidence": min_confidence
                            })
                    
                    self.logger.info(f"Validated {len(structured_validated_skills)} skills, rejected {len(result.get('rejected_skills', []))} skills using model: {model}")
                    return {
                        "validated_skills": structured_validated_skills,
                        "rejected_skills": result.get("rejected_skills", [])
                    }
                except json.JSONDecodeError:
                    self.logger.error(f"Error parsing JSON from OpenAI response")
            
            # If JSON parsing fails, do a simple split of skills
            # Just assume all extracted skills are valid for fallback with default confidence
            self.logger.warning("Failed to parse JSON response, using extracted skills as fallback")
            fallback_skills = [{"name": skill, "confidence": 0.7} for skill in extracted_skills]
            return {"validated_skills": fallback_skills, "rejected_skills": []}
            
        except Exception as e:
            self.logger.error(f"Error validating skills with OpenAI: {str(e)}")
            # Fallback with default confidence
            fallback_skills = [{"name": skill, "confidence": 0.7} for skill in extracted_skills]
            return {"validated_skills": fallback_skills, "rejected_skills": []}
    
    def _mock_extract_skills(self, text: str) -> Dict[str, Any]:
        """
        Mock implementation for testing without API calls.
        
        Args:
            text: Text to extract skills from
            
        Returns:
            Dictionary with mock skill extraction results including confidence scores
        """
        # Example skill keywords to look for
        skill_keywords = [
            "Python", "SQL", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP",
            "Machine Learning", "Data Science", "AI", "Artificial Intelligence",
            "Deep Learning", "NLP", "Natural Language Processing", 
            "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
            "Big Data", "Hadoop", "Spark", "AWS", "Azure", "GCP",
            "Docker", "Kubernetes", "Git", "DevOps", "CI/CD"
        ]
        
        # Lowercase text for case-insensitive matching
        text_lower = text.lower()
        
        # Find matches with mock confidence scores
        validated_skills = []
        for skill in skill_keywords:
            if skill.lower() in text_lower:
                # Add random confidence between 0.7 and 1.0
                import random
                confidence = round(0.7 + random.random() * 0.3, 2)
                validated_skills.append({"name": skill, "confidence": confidence})
        
        # Return mock result
        return {
            "text": text,
            "state": {
                "validated_skills": validated_skills,
                "rejected_skills": ["Coding", "Data"]  # Always reject these generic terms
            }
        }
    
    def extract_skills(self, text: str) -> Dict[str, Any]:
        """
        Extract, validate, and categorize skills from text.
        
        Args:
            text: Text to extract skills from
            
        Returns:
            Dictionary with validated and rejected skills in the required format
        """
        if not self.extraction_config.get("enabled"):
            return {"text": text, "state": {"validated_skills": [], "rejected_skills": []}}
        
        # Use mock implementation if API key not available or method is 'mock'
        if self.extraction_config.get("method") == "mock" or not self.extraction_config.get("openai_api_key"):
            self.logger.info("Using mock skill extraction")
            return self._mock_extract_skills(text)
        
        try:
            # Extract skills
            extraction_result = self._extract_skills_with_openai(text)
            extracted_skills = extraction_result.get("extracted_skills", [])
            
            if not extracted_skills:
                self.logger.info("No skills extracted")
                return {"text": text, "state": {"validated_skills": [], "rejected_skills": []}}
            
            # Validate and categorize skills
            validation_result = self._validate_skills_with_openai(text, extracted_skills)
            
            # Format result
            result = {
                "text": text,
                "state": validation_result
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting skills: {str(e)}")
            return {"text": text, "state": {"validated_skills": [], "rejected_skills": [], "error": str(e)}}
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents and extract skills from their content.
        
        Args:
            documents: List of Document objects to process
            
        Returns:
            List of Document objects with extracted skills in metadata
        """
        if not self.extraction_config.get("enabled"):
            self.logger.info("Skill extraction disabled, returning original documents")
            return documents
        
        self.logger.info(f"Processing {len(documents)} documents for skill extraction")
        processed_documents = []
        
        for doc in documents:
            try:
                # Extract skills from document content
                extraction_result = self.extract_skills(doc.page_content)
                
                # Clone metadata to avoid modifying the original
                metadata = doc.metadata.copy() if doc.metadata else {}
                
                # Add skills to metadata
                metadata["skills"] = extraction_result.get("state", {})
                
                # Create new document with updated metadata
                processed_doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
                
                processed_documents.append(processed_doc)
                
            except Exception as e:
                self.logger.error(f"Error processing document for skill extraction: {str(e)}")
                # Keep original document if processing fails
                processed_documents.append(doc)
        
        self.logger.info(f"Completed skill extraction for {len(processed_documents)} documents")
        return processed_documents 