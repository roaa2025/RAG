"""
Example demonstrating how to directly connect skill extraction with Docling's document processing.
This script shows how to extract documents using Docling and then add skill extraction
data to the processed documents' metadata.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Add parent directory to path so we can import the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocessing.skill_extractor import SkillExtractor
from app.utils.skill_utils import process_skill_data
from langchain.schema.document import Document
from app.utils import document_utils

# Try to import from Docling with different versions
try:
    # Try newer Docling import
    from docling import DocumentProcessor
    from docling.datamodel import DocumentData
    HAS_DOCLING = True
except ImportError:
    try:
        # Try importing from the ocr module directly
        from app.ocr.document_ocr import DocumentOCR
        HAS_DOCLING = False
    except ImportError:
        HAS_DOCLING = False
        print("WARNING: Could not import Docling or DocumentOCR. Will use simple text processing.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_with_docling(file_paths):
    """
    Extract text and metadata from documents using Docling or alternative methods.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        List of extracted documents with metadata
    """
    extracted_docs = []
    
    # Try using DocumentOCR if available
    if not HAS_DOCLING:
        try:
            from app.ocr.document_ocr import DocumentOCR
            extractor = DocumentOCR()
            logger.info("Using DocumentOCR for extraction")
            return extractor.extract_text(file_paths)
        except Exception as e:
            logger.error(f"Error using DocumentOCR: {str(e)}")
            # Continue with fallback method
    
    # Process each file
    for file_path in file_paths:
        try:
            logger.info(f"Processing file: {file_path}")
            file_type = Path(file_path).suffix.lower()
            
            # Different processing based on file type
            if file_type == '.pdf':
                # Try to extract text from PDF
                try:
                    import pypdf
                    with open(file_path, 'rb') as f:
                        pdf = pypdf.PdfReader(f)
                        text = ""
                        for page in pdf.pages:
                            text += page.extract_text() + "\n\n"
                    
                    metadata = {
                        "source": file_path,
                        "file_type": ".pdf",
                        "dl_meta": {"page_count": len(pdf.pages)}
                    }
                    
                    doc = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    extracted_docs.append(doc)
                    logger.info(f"Successfully extracted text from PDF: {file_path}")
                    
                except Exception as pdf_err:
                    logger.error(f"Error extracting text from PDF: {str(pdf_err)}")
                    
            elif file_type in ['.jpg', '.jpeg', '.png', '.webp']:
                # Try to use Tesseract OCR for images
                try:
                    import pytesseract
                    from PIL import Image
                    
                    image = Image.open(file_path)
                    text = pytesseract.image_to_string(image)
                    
                    metadata = {
                        "source": file_path,
                        "file_type": file_type,
                        "dl_meta": {"image_size": image.size}
                    }
                    
                    doc = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    extracted_docs.append(doc)
                    logger.info(f"Successfully extracted text from image: {file_path}")
                    
                except Exception as img_err:
                    logger.error(f"Error extracting text from image: {str(img_err)}")
            
            elif file_type == '.txt':
                # Simple text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                metadata = {
                    "source": file_path,
                    "file_type": ".txt",
                    "dl_meta": {"text_file": True}
                }
                
                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                extracted_docs.append(doc)
                logger.info(f"Successfully read text file: {file_path}")
            
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
    
    return extracted_docs

def process_documents_with_skills(documents, skill_config, output_dir):
    """
    Process documents by extracting skills and consolidating them.
    
    Args:
        documents: List of Document objects
        skill_config: Configuration for skill extraction
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Initialize skill extractor
        logger.info("Initializing skill extractor")
        skill_extractor = SkillExtractor(extraction_config=skill_config)
        
        # Process documents with skill extraction
        logger.info(f"Extracting skills from {len(documents)} documents")
        processed_documents = skill_extractor.process_documents(documents)
        
        # Consolidate skills across all documents
        logger.info("Consolidating skills across all documents")
        skill_data = process_skill_data(
            processed_documents=processed_documents,
            output_dir=output_dir,
            include_rejected=skill_config.get("store_rejected", True)
        )
        
        # Prepare the result
        result = {
            "processed_documents": processed_documents,
            "consolidated_skill_data": skill_data["skill_state"],
            "document_count": len(processed_documents)
        }
        
        # Add paths to generated files
        if "state_path" in skill_data:
            result["skill_state_path"] = skill_data["state_path"]
        if "report_path" in skill_data:
            result["skill_report_path"] = skill_data["report_path"]
        
        # Save the processed documents with skill metadata
        output_path = os.path.join(output_dir, "processed_documents_with_skills.json")
        
        # Convert documents to serializable format
        serializable_docs = []
        for doc in processed_documents:
            doc_dict = {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            serializable_docs.append(doc_dict)
        
        # Save to JSON
        document_utils.save_to_json(
            {
                "processed_documents": serializable_docs,
                "consolidated_skill_data": skill_data["skill_state"]
            }, 
            output_path
        )
        
        result["output_path"] = output_path
        logger.info(f"Saved processed documents with skills to {output_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    """
    Main function to demonstrate integrated skill extraction with Docling.
    """
    # Set the OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY", "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")
    
    # Skill extraction configuration
    skill_config = {
        "enabled": True,
        "openai_api_key": openai_api_key,
        "method": "openai",
        "model": "gpt-4o",
        "min_confidence": 0.5,
        "store_rejected": True
    }
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/docling_skills")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample texts to process - these are more likely to work with Docling
    # Get the image paths from the specified directory
    img_dir = r"C:\Users\roaa.alashqar\Desktop\marketing-rag-docs"
    document_paths = []
    
    
    # If directory doesn't exist, use sample pdf files from examples directory
    if not os.path.exists(img_dir):
        logger.warning(f"Directory not found: {img_dir}")
        logger.info("Using sample text files instead")
        
        # Create temporary text files
        sample_texts = [
            # Sample 1: Software Developer Resume
            """
            John Smith
            Software Developer
            
            Skills: Python, SQL, and machine learning
            Experience with natural language processing and deep learning frameworks like TensorFlow and PyTorch.
            Knowledge of big data technologies such as Hadoop and Spark.
            """,
            
            # Sample 2: Data Scientist Resume
            """
            Maria Rodriguez
            Data Scientist
            
            Expertise in statistical analysis, R programming, and Python
            Experience with scikit-learn, TensorFlow, and Keras
            Proficient in data visualization tools like Tableau and PowerBI
            """
        ]
        
        temp_files = []
        for i, text in enumerate(sample_texts):
            try:
                # Try creating PDF files
                temp_file_path = os.path.join(output_dir, f"sample_document_{i+1}.pdf")
                try:
                    from fpdf import FPDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    
                    # Add text to PDF
                    for line in text.split('\n'):
                        pdf.cell(200, 10, txt=line, ln=True)
                    
                    pdf.output(temp_file_path)
                    temp_files.append(temp_file_path)
                    logger.info(f"Created temporary PDF file: {temp_file_path}")
                except Exception as e:
                    logger.error(f"Error creating PDF file: {str(e)}")
                    
                    # Fall back to text file if PDF creation fails
                    temp_file_path = os.path.join(output_dir, f"sample_document_{i+1}.txt")
                    with open(temp_file_path, "w") as f:
                        f.write(text)
                    temp_files.append(temp_file_path)
                    logger.info(f"Created temporary text file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error creating sample document: {str(e)}")
        
        document_paths = temp_files
    else:
        # Use the specified image directory
        for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.webp']:
            import glob
            document_paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    
    if not document_paths:
        logger.error("No documents found to process")
        print("No documents found to process. Please provide valid document paths.")
        return
    
    print(f"Found {len(document_paths)} documents to process")
    
    try:
        # Step 1: Extract text from documents using Docling
        print("\nStep 1: Extracting text from documents using Docling or fallback methods")
        print("=" * 70)
        extracted_docs = extract_text_with_docling(document_paths)
        
        if not extracted_docs:
            logger.error("No documents were successfully extracted")
            print("No documents were successfully extracted. Please check the logs for details.")
            return
        
        print(f"Successfully extracted {len(extracted_docs)} documents")
        
        # Step 2: Process documents with skill extraction
        print("\nStep 2: Processing documents with skill extraction")
        print("=" * 50)
        result = process_documents_with_skills(extracted_docs, skill_config, output_dir)
        
        if "error" in result:
            logger.error("Error during processing")
            print(f"Error during processing: {result['error']}")
            return
        
        print(f"Successfully processed {result['document_count']} documents")
        
        # Step 3: Display consolidated skill results
        if "consolidated_skill_data" in result:
            print("\nStep 3: Displaying consolidated skill results")
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
        
        # Print the output file path
        if "output_path" in result:
            print(f"\nOutput saved to: {result['output_path']}")
            print("This file contains the processed documents with skill metadata.")
        
        # Print paths to skill reports
        if "skill_state_path" in result:
            print(f"Skill state JSON saved to: {result['skill_state_path']}")
        if "skill_report_path" in result:
            print(f"Detailed skill report saved to: {result['skill_report_path']}")
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 