"""
Utility functions for working with extracted skills data.

This module provides functions for consolidating skills extracted from
multiple documents and generating skill reports.
"""

from typing import Dict, List, Any, Set, Optional, Union, Tuple
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def consolidate_skills(processed_documents: List[Any]) -> Dict[str, Any]:
    """
    Consolidate all skills extracted from multiple documents into a single state.
    
    Args:
        processed_documents: List of document dictionaries or Document objects with skill extraction metadata
        
    Returns:
        Dictionary with consolidated validated and rejected skills
    """
    validated_skills: Dict[str, Dict[str, Union[float, int]]] = {}
    all_rejected_skills: Set[str] = set()
    document_skill_mapping: Dict[str, Dict[str, Any]] = {}
    
    for doc in processed_documents:
        # Handle both dictionary and Document objects
        if hasattr(doc, 'metadata'):
            # LangChain Document object
            metadata = doc.metadata
            # Check for the new skills field first, then fall back to skill_extraction
            skill_data = metadata.get("skills", metadata.get("skill_extraction", {}))
            doc_id = metadata.get("source", "unknown")
        else:
            # Dictionary object
            if not doc.get("metadata"):
                continue
                
            metadata = doc.get("metadata", {})
            # First try to get skills from the new location
            skill_data = metadata.get("skills", metadata.get("skill_extraction", {}))
            doc_id = metadata.get("source", "unknown")
        
        # Add validated skills
        if skill_data and "validated_skills" in skill_data:
            validated_skills_list = skill_data["validated_skills"]
            
            # Map skills to document
            if doc_id not in document_skill_mapping:
                document_skill_mapping[doc_id] = {"validated": [], "rejected": []}
                
            # Handle both old format (list of strings) and new format (list of dicts with name and confidence)
            doc_validated_skills = []
            
            for skill_item in validated_skills_list:
                if isinstance(skill_item, dict):
                    # New format with confidence scores
                    skill_name = skill_item.get("name")
                    confidence = skill_item.get("confidence", 0.7)
                    
                    if skill_name:
                        doc_validated_skills.append({
                            "name": skill_name,
                            "confidence": confidence
                        })
                        
                        # Update global validated skills dictionary
                        if skill_name not in validated_skills:
                            validated_skills[skill_name] = {
                                "count": 0,
                                "total_confidence": 0,
                                "avg_confidence": 0
                            }
                            
                        validated_skills[skill_name]["count"] += 1
                        validated_skills[skill_name]["total_confidence"] += confidence
                        validated_skills[skill_name]["avg_confidence"] = (
                            validated_skills[skill_name]["total_confidence"] / 
                            validated_skills[skill_name]["count"]
                        )
                else:
                    # Old format (just strings)
                    skill_name = skill_item
                    doc_validated_skills.append({
                        "name": skill_name,
                        "confidence": 0.7  # Default confidence
                    })
                    
                    # Update global validated skills dictionary
                    if skill_name not in validated_skills:
                        validated_skills[skill_name] = {
                            "count": 0,
                            "total_confidence": 0,
                            "avg_confidence": 0
                        }
                        
                    validated_skills[skill_name]["count"] += 1
                    validated_skills[skill_name]["total_confidence"] += 0.7
                    validated_skills[skill_name]["avg_confidence"] = (
                        validated_skills[skill_name]["total_confidence"] / 
                        validated_skills[skill_name]["count"]
                    )
            
            document_skill_mapping[doc_id]["validated"] = doc_validated_skills
        
        # Add rejected skills if available
        if skill_data and "rejected_skills" in skill_data:
            rejected_skills_list = skill_data["rejected_skills"]
            all_rejected_skills.update(rejected_skills_list)
            
            if doc_id not in document_skill_mapping:
                document_skill_mapping[doc_id] = {"validated": [], "rejected": []}
                
            document_skill_mapping[doc_id]["rejected"] = rejected_skills_list
    
    # Convert validated skills dictionary to list format with confidence
    validated_skills_list = [
        {
            "name": skill_name,
            "confidence": round(data["avg_confidence"], 2),
            "count": data["count"]
        }
        for skill_name, data in validated_skills.items()
    ]
    
    # Sort by confidence and count
    validated_skills_list.sort(key=lambda x: (x["confidence"], x["count"]), reverse=True)
    
    # Convert rejected skills set to sorted list
    rejected_skills_list = sorted(list(all_rejected_skills))
    
    # Create consolidated state
    consolidated_state = {
        "validated_skills": validated_skills_list,
        "rejected_skills": rejected_skills_list,
        "document_skill_mapping": document_skill_mapping
    }
    
    return consolidated_state

def calculate_skill_frequencies(consolidated_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate frequency statistics for validated skills across documents.
    
    Args:
        consolidated_state: The consolidated skills state from multiple documents
        
    Returns:
        Dictionary with skill frequency data added
    """
    # Create new dictionary for frequencies
    skill_frequencies = {}
    
    # Get validated skills - already includes count and confidence
    validated_skills = consolidated_state.get("validated_skills", [])
    
    for skill_obj in validated_skills:
        if isinstance(skill_obj, dict):
            skill_name = skill_obj.get("name")
            count = skill_obj.get("count", 1)
            skill_frequencies[skill_name] = count
        else:
            # Handle legacy format (string only)
            skill_frequencies[skill_obj] = 1
    
    # Sort skills by frequency (descending)
    sorted_skills = sorted(
        skill_frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Add frequency data to the state
    result = consolidated_state.copy()
    result["skill_frequencies"] = {
        skill: count for skill, count in sorted_skills
    }
    
    # Add top skills with confidence scores
    result["top_skills"] = []
    for skill_name, _ in sorted_skills[:10]:
        # Find the skill object in validated_skills
        skill_obj = next((s for s in validated_skills if s.get("name") == skill_name), None)
        if skill_obj:
            result["top_skills"].append(skill_obj)
        else:
            # Fallback
            result["top_skills"].append({"name": skill_name, "confidence": 0.7, "count": 1})
    
    return result

def generate_skill_report(
    consolidated_state: Dict[str, Any],
    include_rejected: bool = False
) -> str:
    """
    Generate a human-readable report of extracted skills.
    
    Args:
        consolidated_state: The consolidated skills state from multiple documents
        include_rejected: Whether to include rejected skills in the report
        
    Returns:
        A formatted string with the skill report
    """
    report = []
    report.append("SKILL EXTRACTION REPORT")
    report.append("=====================")
    report.append("")
    
    # Summary
    validated_count = len(consolidated_state.get("validated_skills", []))
    rejected_count = len(consolidated_state.get("rejected_skills", []))
    doc_count = len(consolidated_state.get("document_skill_mapping", {}))
    
    report.append(f"Documents processed: {doc_count}")
    report.append(f"Total validated skills: {validated_count}")
    report.append(f"Total rejected skills: {rejected_count}")
    report.append("")
    
    # Top skills by frequency
    if "top_skills" in consolidated_state and consolidated_state["top_skills"]:
        report.append("TOP SKILLS")
        report.append("---------")
        for skill_obj in consolidated_state["top_skills"]:
            if isinstance(skill_obj, dict):
                skill_name = skill_obj.get("name", "")
                confidence = skill_obj.get("confidence", 0)
                count = skill_obj.get("count", 0)
                report.append(f"- {skill_name} (confidence: {confidence:.2f}, found in {count} document{'s' if count != 1 else ''})")
            else:
                # Legacy format
                skill = skill_obj
                frequency = consolidated_state.get("skill_frequencies", {}).get(skill, 0)
                report.append(f"- {skill} (found in {frequency} document{'s' if frequency != 1 else ''})")
        report.append("")
    
    # All validated skills
    report.append("ALL VALIDATED SKILLS")
    report.append("-------------------")
    for skill_obj in consolidated_state.get("validated_skills", []):
        if isinstance(skill_obj, dict):
            skill_name = skill_obj.get("name", "")
            confidence = skill_obj.get("confidence", 0)
            count = skill_obj.get("count", 0)
            report.append(f"- {skill_name} (confidence: {confidence:.2f}, found in {count} document{'s' if count != 1 else ''})")
        else:
            # Legacy format
            skill = skill_obj
            frequency = consolidated_state.get("skill_frequencies", {}).get(skill, 0)
            report.append(f"- {skill} (found in {frequency} document{'s' if frequency != 1 else ''})")
    report.append("")
    
    # Rejected skills
    if include_rejected and consolidated_state.get("rejected_skills"):
        report.append("REJECTED SKILLS")
        report.append("--------------")
        for skill in consolidated_state.get("rejected_skills", []):
            report.append(f"- {skill}")
        report.append("")
    
    # Skills by document
    report.append("SKILLS BY DOCUMENT")
    report.append("-----------------")
    for doc_id, skills in consolidated_state.get("document_skill_mapping", {}).items():
        report.append(f"Document: {doc_id}")
        
        if skills.get("validated"):
            report.append("  Validated skills:")
            for skill_item in skills["validated"]:
                if isinstance(skill_item, dict):
                    skill_name = skill_item.get("name", "")
                    confidence = skill_item.get("confidence", 0)
                    report.append(f"  - {skill_name} (confidence: {confidence:.2f})")
                else:
                    # Legacy format
                    report.append(f"  - {skill_item}")
        else:
            report.append("  No validated skills")
            
        if include_rejected and skills.get("rejected"):
            report.append("  Rejected skills:")
            for skill in skills["rejected"]:
                report.append(f"  - {skill}")
        
        report.append("")
    
    return "\n".join(report)

def save_skill_state(
    consolidated_state: Dict[str, Any],
    output_dir: str,
    filename: str = "skill_extraction_state.json"
) -> str:
    """
    Save the consolidated skill state to a JSON file.
    
    Args:
        consolidated_state: The consolidated skills state
        output_dir: Directory to save the file
        filename: Name of the output file
        
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(consolidated_state, f, indent=2)
        logger.info(f"Saved skill state to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving skill state: {str(e)}")
        return ""

def save_skill_report(
    report: str,
    output_dir: str,
    filename: str = "skill_extraction_report.txt"
) -> str:
    """
    Save the skill report to a text file.
    
    Args:
        report: The formatted report string
        output_dir: Directory to save the file
        filename: Name of the output file
        
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved skill report to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving skill report: {str(e)}")
        return ""

def process_skill_data(
    processed_documents: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    include_rejected: bool = True
) -> Dict[str, Any]:
    """
    Process document data to extract a consolidated skill state and generate reports.
    
    Args:
        processed_documents: List of document dictionaries with skill extraction metadata
        output_dir: Optional directory to save output files
        include_rejected: Whether to include rejected skills in the report
        
    Returns:
        Dictionary with results including consolidated state and report file paths
    """
    # Ensure documents are in the correct format
    formatted_documents = []
    for doc in processed_documents:
        if isinstance(doc, str):
            # If it's a string, create a basic document structure
            formatted_doc = {
                "text": doc,
                "metadata": {
                    "source": "unknown",
                    "skill_extraction": {
                        "validated_skills": [],
                        "rejected_skills": []
                    }
                }
            }
            formatted_documents.append(formatted_doc)
        elif isinstance(doc, dict):
            # If it's already a dictionary, ensure it has the required structure
            if "metadata" not in doc:
                doc["metadata"] = {}
            if "skill_extraction" not in doc["metadata"]:
                doc["metadata"]["skill_extraction"] = {
                    "validated_skills": [],
                    "rejected_skills": []
                }
            formatted_documents.append(doc)
        else:
            # For other types, try to convert to dict
            try:
                doc_dict = {
                    "text": str(doc),
                    "metadata": {
                        "source": "unknown",
                        "skill_extraction": {
                            "validated_skills": [],
                            "rejected_skills": []
                        }
                    }
                }
                formatted_documents.append(doc_dict)
            except Exception as e:
                logger.warning(f"Could not process document: {str(e)}")
                continue
    
    # Consolidate skills from all documents
    consolidated_state = consolidate_skills(formatted_documents)
    
    # Calculate skill frequencies
    consolidated_state = calculate_skill_frequencies(consolidated_state)
    
    # Generate report
    skill_report = generate_skill_report(consolidated_state, include_rejected=include_rejected)
    
    result = {
        "skill_state": consolidated_state,
        "skill_report": skill_report
    }
    
    # Save files if output directory is provided
    if output_dir:
        state_path = save_skill_state(consolidated_state, output_dir)
        report_path = save_skill_report(skill_report, output_dir)
        
        if state_path:
            result["state_path"] = state_path
        if report_path:
            result["report_path"] = report_path
    
    return result 