from typing import Dict, List, Any, Optional, Union, cast
import logging
import os
import json
from pathlib import Path
import uuid

from app.ocr.document_ocr import DocumentOCR
from app.preprocessing.text_preprocessor import TextPreprocessor
from app.preprocessing.skill_extractor import SkillExtractor
from app.structuring.data_structurer import DataStructurer
from app.structuring.advanced_data_structurer import AdvancedDataStructurer
from app.embedding_service import EmbeddingService
from app.utils import document_utils
from app.utils.config import get_default_config, merge_configs
from app.utils.skill_utils import process_skill_data
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DocumentParser:
    """
    Main class for document parsing, integrating OCR extraction, text preprocessing,
    and data structuring components.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the document parser with optional configuration.
        
        Args:
            config: Configuration dictionary for the parser components
        """
        # Load default config and merge with user config
        default_config = get_default_config()
        self.config = merge_configs(default_config, config or {})
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DocumentParser with merged configuration")
        
        # Initialize components with their respective configurations
        self.ocr_extractor = DocumentOCR(
            ocr_config=self.config.get("ocr", {})
        )
        
        self.text_preprocessor = TextPreprocessor(
            preprocessing_config=self.config.get("preprocessing", {})
        )
        
        # Initialize skill extractor if enabled
        self.skill_extractor = None

        if self.config.get("skill_extraction", {}).get("enabled", False):
            self.logger.info("Initializing skill extractor")
            self.skill_extractor = SkillExtractor(
                extraction_config=self.config.get("skill_extraction", {})
            )
        
        # Initialize embedding service if enabled
        self.embedding_service = None
        if self.config.get("embeddings", {}).get("enabled", False):
            self.logger.info("Initializing embedding service")
            self.embedding_service = EmbeddingService()
        
        # Initialize data structurer
        self.data_structurer = self._initialize_data_structurer()
        
        self.logger.info("DocumentParser initialization complete")
    
    def _initialize_data_structurer(self):
        """
        Initialize the appropriate data structurer based on configuration.
        
        Returns:
            Either AdvancedDataStructurer or DataStructurer instance
        """
        # Use advanced data structurer if enabled in config
        use_advanced_chunking = self.config.get("structuring", {}).get("use_advanced_chunking", False)
        
        # Ensure structuring config exists
        if "structuring" not in self.config:
            self.config["structuring"] = {}
        
        # Set advanced chunking flag
        self.config["structuring"]["use_advanced_chunking"] = use_advanced_chunking
        
        if use_advanced_chunking:
            self.logger.info("Using advanced chunking strategy")
            return AdvancedDataStructurer(
                structure_config=self.config.get("structuring", {})
            )
        else:
            self.logger.info("Using basic document structuring")
            return DataStructurer(
                structure_config=self.config.get("structuring", {})
            )
    
    def parse_document(self, file_path: Union[str, List[str]], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse a document or list of documents through the complete pipeline:
        OCR extraction -> Text preprocessing -> Skill extraction -> Data structuring.
        
        Args:
            file_path: Path to the document or list of document paths
            output_dir: Optional directory to save the output
            
        Returns:
            Dictionary with parsing results
        """
        self.logger.info(f"Starting document parsing for: {file_path}")
        
        # Validate input file paths
        valid_paths = document_utils.ensure_valid_input(file_path)
        self.logger.info(f"Processing {len(valid_paths)} valid document(s)")
        
        # Step 1: OCR Extraction
        self.logger.info("Step 1: OCR Extraction")
        extracted_documents = self.ocr_extractor.extract_text(valid_paths)
        self.logger.info(f"Extracted {len(extracted_documents)} document chunks")
        
        # Step 2: Text Preprocessing
        self.logger.info("Step 2: Text Preprocessing")
        processed_documents = self.text_preprocessor.process_documents(extracted_documents)
        self.logger.info(f"Preprocessed {len(processed_documents)} document chunks")
        
        # Step 2.5: Skill Extraction (if enabled)
        if self.skill_extractor:
            self.logger.info("Step 2.5: Skill Extraction")
            processed_documents = self.skill_extractor.process_documents(processed_documents)
            self.logger.info(f"Extracted skills from {len(processed_documents)} document chunks")
        
        # Step 3: Data Structuring with Chunking
        self.logger.info("Step 3: Data Structuring and Chunking")
        structured_documents = self.data_structurer.structure_documents(processed_documents)
        
        # Check if we're using the advanced structurer
        if isinstance(self.data_structurer, AdvancedDataStructurer):
            structured_docs = cast(Dict[str, Any], structured_documents)
            self.logger.info(f"Generated {structured_docs.get('total_chunk_count', 0)} chunks across {len(processed_documents)} documents")
            result = {
                "input_files": valid_paths,
                "document_count": len(valid_paths),
                **structured_docs  # Include all evaluation data
            }
        else:
            self.logger.info(f"Structured {len(structured_documents)} document chunks")
            result = {
                "input_files": valid_paths,
                "document_count": len(valid_paths),
                "chunk_count": len(structured_documents),
                "processed_documents": structured_documents
            }

        # Step 4: Consolidate skills across all documents (if skill extraction is enabled)
        if self.skill_extractor and output_dir:
            self.logger.info("Step 4: Consolidating skills across documents")
            
            # Handle both advanced and basic structuring formats
            if isinstance(self.data_structurer, AdvancedDataStructurer):
                # For advanced structuring, we need to extract skills from both documents and chunks
                docs_to_process: List[Dict[str, Any]] = []
                
                # First, add the original documents 
                if "documents" in structured_docs:
                    docs_to_process.extend(cast(List[Dict[str, Any]], structured_docs["documents"]))
                
                # Then add individual chunks which might contain skills
                if "chunks" in structured_docs:
                    # Convert chunks to the format expected by process_skill_data
                    for chunk in cast(List[Dict[str, Any]], structured_docs["chunks"]):
                        if isinstance(chunk, dict):
                            chunk_doc = {
                                "content_type": chunk.get("content_type", "text"),
                                "metadata": chunk.get("metadata", {}),
                                "raw_text": chunk.get("content", "")
                            }
                            docs_to_process.append(chunk_doc)
            else:
                # For basic structuring, use the processed_documents directly
                docs_to_process = cast(List[Dict[str, Any]], structured_documents)
            
            # Process skill data from all documents
            include_rejected = self.config.get("skill_extraction", {}).get("store_rejected", True)
            skill_data = process_skill_data(
                processed_documents=docs_to_process,
                output_dir=output_dir,
                include_rejected=include_rejected
            )
            
            # Add consolidated skill data to the result
            result["consolidated_skill_data"] = skill_data.get("skill_state", {})
            
            # Add paths to generated files
            if "state_path" in skill_data:
                result["skill_state_path"] = skill_data["state_path"]
            if "report_path" in skill_data:
                result["skill_report_path"] = skill_data["report_path"]
            
            self.logger.info(f"Generated consolidated skill state with {len(skill_data.get('skill_state', {}).get('validated_skills', []))} skills")
        
        # Step 5: Create embeddings if enabled
        if self.embedding_service and output_dir:
            self.logger.info("Step 5: Creating embeddings for document chunks")
            
            # Save result to a temporary file if not already saved
            if "output_path" not in result:
                # Create a temporary file path
                if len(valid_paths) == 1:
                    # Single document
                    tmp_output_path = document_utils.create_output_filename(
                        valid_paths[0], 
                        output_dir=output_dir,
                        suffix="_temp"
                    )
                else:
                    # Multiple documents
                    tmp_output_path = os.path.join(output_dir, "temp_parsed_documents.json")
                    
                document_utils.save_to_json(result, tmp_output_path)
                json_path = tmp_output_path
            else:
                json_path = result["output_path"]
            
            # Process the JSON file to create embeddings
            try:
                embeddings_result = self.embedding_service.process_json_file(json_path)
                
                # Save embeddings
                embeddings_output = os.path.join(output_dir, "document_embeddings.json")
                self.embedding_service.save_embeddings(embeddings_result, embeddings_output)
                
                # Add embeddings information to the result
                result["embeddings_path"] = embeddings_output
                result["embedding_count"] = len(embeddings_result.get("processed_chunks", []))
                
                self.logger.info(f"Created {result['embedding_count']} embeddings")
                
                # Clean up temporary file if it was created
                if "output_path" not in result and os.path.exists(tmp_output_path):
                    os.remove(tmp_output_path)
                    
            except Exception as e:
                self.logger.error(f"Error creating embeddings: {str(e)}")
                # Don't fail the entire pipeline if embeddings fail
                result["embedding_error"] = str(e)
        
        # Save result to file if output_dir is provided
        if output_dir and "output_path" not in result:
            if len(valid_paths) == 1:
                # Single document
                output_path = document_utils.create_output_filename(
                    valid_paths[0], 
                    output_dir=output_dir
                )
            else:
                # Multiple documents
                output_path = os.path.join(output_dir, "parsed_documents.json")
                
            document_utils.save_to_json(result, output_path)
            self.logger.info(f"Results saved to {output_path}")
            result["output_path"] = output_path
            
        return result

    def process_documents(self, document_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple documents and create embeddings.
        
        Args:
            document_paths: List of paths to documents to process
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "document_count": len(document_paths),
            "processed_documents": [],
            "total_chunk_count": 0
        }
        
        # Process each document
        for doc_path in document_paths:
            try:
                # Process document and get chunks
                doc_result = self.parse_document(doc_path)
                
                if doc_result:
                    # Add document to results
                    result["processed_documents"].append(doc_result)
                    result["total_chunk_count"] += len(doc_result.get("chunks", []))
                    
                    # Store embeddings in Qdrant if embedding service is available
                    if self.embedding_service and doc_result.get("chunks"):
                        try:
                            # Create Document objects from chunks
                            documents = [
                                Document(
                                    page_content=chunk["text"],
                                    metadata={
                                        "doc_id": doc_result.get("id", str(uuid.uuid4())),
                                        "source": doc_path,
                                        "chunk_index": i,
                                        "chunk_id": f"chunk_{i}",
                                        "document_title": os.path.basename(doc_path),
                                        "content_type": chunk.get("content_type", "text"),
                                        **chunk.get("metadata", {})
                                    }
                                )
                                for i, chunk in enumerate(doc_result["chunks"])
                            ]
                            
                            # Store documents in Qdrant
                            doc_ids = self.embedding_service.store_documents(documents)
                            doc_result["embedding_ids"] = doc_ids
                            
                        except Exception as e:
                            self.logger.error(f"Error storing embeddings for {doc_path}: {str(e)}")
                            doc_result["embedding_error"] = str(e)
                            
            except Exception as e:
                self.logger.error(f"Error processing document {doc_path}: {str(e)}")
                result["processed_documents"].append({
                    "id": str(uuid.uuid4()),
                    "path": doc_path,
                    "error": str(e)
                })
        
        # Save results if output directory is provided
        if output_dir:
            output_path = os.path.join(output_dir, "processed_documents.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            result["output_path"] = output_path
            
        return result

def parse_document(file_path: Union[str, List[str]], config: Optional[Dict] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Utility function for parsing documents without instantiating the DocumentParser class.
    
    Args:
        file_path: Path to the document or list of document paths
        config: Configuration dictionary for the parser components
        output_dir: Optional directory to save the output
        
    Returns:
        Dictionary with parsing results
    """
    parser = DocumentParser(config=config)
    return parser.parse_document(file_path, output_dir=output_dir)

def run_complete_pipeline(file_paths: Union[str, List[str]], 
                         output_dir: str = "output",
                         config_path: Optional[str] = None,
                         store_in_qdrant: bool = True,
                         qdrant_collection: str = "document_embeddings") -> Dict[str, Any]:
    """
    Run the complete document processing pipeline from document extraction to 
    storage in Qdrant vector store.
    
    Pipeline steps:
    1. Extract text from documents (OCR if needed)
    2. Preprocess the text
    3. Extract skills from the text
    4. Apply advanced chunking
    5. Generate embeddings for chunks
    6. Store document chunks in Qdrant vector store
    
    Args:
        file_paths: Path to the document(s) to process
        output_dir: Directory to save output files
        config_path: Path to custom config file (optional)
        store_in_qdrant: Whether to store results in Qdrant
        qdrant_collection: Name of the Qdrant collection to use
        
    Returns:
        Dictionary with results from the complete pipeline
    """
    from qdrant_client import QdrantClient
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load custom configuration if provided
    config = None
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Ensure configuration has necessary components enabled
    if not config:
        config = {
            "ocr": {"enabled": True},
            "preprocessing": {"enabled": True},
            "skill_extraction": {"enabled": True},
            "structuring": {
                "use_advanced_chunking": True,
                "chunking": {
                    "hierarchical": {"enabled": True},
                    "semantic": {"enabled": True},
                    "fixed_size": {"enabled": True},
                    "metadata": {
                        "preserve_headers": True,
                        "include_position": True,
                        "include_chunk_type": True,
                        "track_pipeline": True
                    }
                }
            },
            "embeddings": {"enabled": True}
        }
    
    # 1. Initialize the document parser with advanced configuration
    logger.info("Initializing document parser with advanced configuration")
    parser = DocumentParser(config=config)
    
    # 2. Parse documents through the pipeline (OCR → preprocessing → skill extraction → structuring)
    logger.info(f"Starting document parsing for {file_paths}")
    parse_results = parser.parse_document(file_paths, output_dir=output_dir)
    
    # 3. Process embeddings 
    embeddings_path = os.path.join(output_dir, "document_embeddings.json")
    
    # Check if embeddings were created as part of the parsing process
    if "embeddings_path" in parse_results:
        logger.info(f"Embeddings were generated in: {parse_results['embeddings_path']}")
        embeddings_path = parse_results["embeddings_path"]
    
    # 4. Store embeddings in Qdrant if requested
    if store_in_qdrant:
        try:
            logger.info(f"Connecting to Qdrant and storing embeddings from {embeddings_path}")
            
            # Initialize Qdrant client
            qdrant_client = QdrantClient("localhost", port=6333)
            
            # Initialize embedding service with client
            embedding_service = EmbeddingService(qdrant_client=qdrant_client)
            
            # Load embeddings from the generated JSON file and store in Qdrant
            qdrant_results = embedding_service.load_and_store_embeddings(
                json_file_path=embeddings_path,
                collection_name=qdrant_collection,
                batch_size=100
            )
            
            # Add Qdrant storage information to results
            parse_results["qdrant_storage"] = {
                "collection_name": qdrant_collection,
                "embeddings_stored": qdrant_results.get("embeddings_inserted", 0),
                "elapsed_seconds": qdrant_results.get("elapsed_seconds", 0),
            }
            
            logger.info(f"Successfully stored {qdrant_results.get('embeddings_inserted', 0)} embeddings in Qdrant")
        
        except Exception as e:
            logger.error(f"Error storing embeddings in Qdrant: {str(e)}")
            parse_results["qdrant_error"] = str(e)
    
    # Update results path if it exists
    result_path = os.path.join(output_dir, "pipeline_results.json")
    try:
        with open(result_path, 'w') as f:
            json.dump(parse_results, f, indent=2)
        logger.info(f"Complete pipeline results saved to {result_path}")
    except Exception as e:
        logger.error(f"Error saving pipeline results: {str(e)}")
    
    return parse_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the complete document processing pipeline")
    parser.add_argument("file_paths", nargs="+", help="Path(s) to document(s) to process")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--no-qdrant-store", action="store_true", help="Skip storing in Qdrant")
    parser.add_argument("--collection", default="document_embeddings", help="Qdrant collection name")
    
    # For debugging purposes, you can uncomment and modify these lines:
    # import sys
    # if len(sys.argv) == 1:  # No arguments provided
    #     sys.argv.extend([
    #         "path/to/your/document.pdf",  # Replace with your document path
    #         "--output-dir", "output"
    #     ])
    
    args = parser.parse_args()
    
    run_complete_pipeline(
        file_paths=args.file_paths,
        output_dir=args.output_dir,
        config_path=args.config,
        store_in_qdrant=not args.no_qdrant_store,
        qdrant_collection=args.collection
    ) 