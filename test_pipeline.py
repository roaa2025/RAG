import os
import sys
import unittest
import tempfile
from pathlib import Path
import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Add parent directory to path so we can import the app
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.main import DocumentParser
from app.ocr.document_ocr import DocumentOCR
from app.preprocessing.text_preprocessor import TextPreprocessor
from app.preprocessing.skill_extractor import SkillExtractor
from app.structuring.data_structurer import DataStructurer, AdvancedDataStructurer
from app.embedding_service import EmbeddingService

class TestDocumentPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test documents
        self.create_test_documents()
        
        # Initialize parser with test configuration
        self.config = {
            "ocr": {
                "ocr_lang": "eng",
                "ocr_dpi": 300,
                "force_ocr": True
            },
            "preprocessing": {
                "normalization": {
                    "lowercase": True,
                    "remove_extra_whitespace": True
                }
            },
            "skill_extraction": {
                "enabled": True,
                "store_rejected": True,
                "min_confidence": 0.7,
                "skills_database": {
                    "technical_skills": ["Python", "Machine Learning", "Data Analysis", "SQL", "AWS"],
                    "soft_skills": ["Leadership", "Communication", "Problem Solving"]
                }
            },
            "structuring": {
                "use_advanced_chunking": True,
                "chunking": {
                    "fixed_size": {
                        "max_tokens": 1500,
                        "overlap_tokens": 150
                    },
                    "semantic": {
                        "min_similarity": 0.7,
                        "max_chunk_size": 2000
                    },
                    "hierarchical": {
                        "max_depth": 3,
                        "min_section_size": 500
                    }
                }
            },
            "embeddings": {
                "enabled": True,
                "model": "text-embedding-ada-002",
                "vector_store": {
                    "collection_name": "test_documents",
                    "distance_metric": "cosine"
                }
            }
        }
        self.parser = DocumentParser(config=self.config)
        
        # Initialize Qdrant client for testing
        self.qdrant_client = QdrantClient(":memory:")  # Use in-memory storage for testing

    def create_test_documents(self):
        """Create test documents in the temporary directory."""
        # Create a text file with structured content
        self.text_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.text_file, "w", encoding="utf-8") as f:
            f.write("""# Professional Resume

## Technical Skills
The candidate has extensive experience with Python, Machine Learning, and Data Analysis. 
They are proficient in SQL and AWS cloud services.

## Soft Skills
Strong leadership abilities and excellent communication skills. 
Demonstrated problem-solving capabilities in various projects.

## Experience
Led multiple data science projects using advanced machine learning techniques.
Implemented scalable solutions using AWS services.
""")

        # Create a PDF file with similar content
        self.pdf_file = os.path.join(self.temp_dir, "test.pdf")
        # You can copy an existing PDF file here or create a new one

    def test_ocr_extraction(self):
        """Test the OCR extraction step."""
        # Initialize OCR extractor
        ocr_extractor = DocumentOCR(ocr_config=self.config.get("ocr", {}))
        
        # Test text file extraction
        documents = ocr_extractor.extract_text(self.text_file)
        self.assertIsInstance(documents, list)
        self.assertTrue(len(documents) > 0)
        
        # Verify document structure
        doc = documents[0]
        self.assertIn("page_content", doc.__dict__)
        self.assertIn("metadata", doc.__dict__)
        self.assertIn("source", doc.metadata)
        self.assertIn("file_type", doc.metadata)

    def test_text_preprocessing(self):
        """Test the text preprocessing step."""
        # Initialize preprocessor
        preprocessor = TextPreprocessor(preprocessing_config=self.config.get("preprocessing", {}))
        
        # Create a test document
        from langchain.schema.document import Document
        test_doc = Document(
            page_content="  This is a TEST document with   extra spaces  ",
            metadata={"source": "test.txt"}
        )
        
        # Process the document
        processed_docs = preprocessor.process_documents([test_doc])
        
        # Verify preprocessing results
        self.assertEqual(len(processed_docs), 1)
        processed_text = processed_docs[0].page_content
        self.assertEqual(processed_text, "this is a test document with extra spaces")

    def test_skill_extraction(self):
        """Test the skill extraction step."""
        # Initialize skill extractor
        skill_extractor = SkillExtractor(extraction_config=self.config.get("skill_extraction", {}))
        
        # Create a test document with skills
        from langchain.schema.document import Document
        test_doc = Document(
            page_content="The candidate has experience with Python, Machine Learning, and Data Analysis.",
            metadata={"source": "test.txt"}
        )
        
        # Process the document
        processed_docs = skill_extractor.process_documents([test_doc])
        
        # Verify skill extraction results
        self.assertEqual(len(processed_docs), 1)
        doc_metadata = processed_docs[0].metadata
        self.assertIn("skill_extraction", doc_metadata)
        self.assertIn("validated_skills", doc_metadata["skill_extraction"])

    def test_data_structuring(self):
        """Test the data structuring step."""
        # Initialize structurer
        structurer = DataStructurer(structure_config=self.config.get("structuring", {}))
        
        # Create a test document with structured content
        from langchain.schema.document import Document
        test_doc = Document(
            page_content="Header\n\nThis is the main content.\n\nFooter",
            metadata={"source": "test.txt"}
        )
        
        # Process the document
        structured_docs = structurer.structure_documents([test_doc])
        
        # Verify structuring results
        self.assertTrue(len(structured_docs) > 0)
        self.assertIn("content_type", structured_docs[0])
        self.assertIn("structured_content", structured_docs[0])

    def test_advanced_chunking(self):
        """Test the advanced chunking functionality."""
        # Initialize advanced structurer
        structurer = AdvancedDataStructurer(structure_config=self.config.get("structuring", {}))
        
        # Create a test document with structured content
        from langchain.schema.document import Document
        test_doc = Document(
            page_content=self.text_file,
            metadata={"source": "test.txt"}
        )
        
        # Process the document
        structured_docs = structurer.structure_documents([test_doc])
        
        # Verify advanced chunking results
        self.assertTrue(len(structured_docs) > 0)
        self.assertIn("chunk_type", structured_docs[0])
        self.assertIn("hierarchical_level", structured_docs[0])
        self.assertIn("semantic_similarity", structured_docs[0])

    def test_skill_extraction_with_confidence(self):
        """Test the skill extraction with confidence scoring."""
        # Initialize skill extractor with confidence threshold
        skill_extractor = SkillExtractor(extraction_config=self.config.get("skill_extraction", {}))
        
        # Create a test document with skills
        from langchain.schema.document import Document
        test_doc = Document(
            page_content=self.text_file,
            metadata={"source": "test.txt"}
        )
        
        # Process the document
        processed_docs = skill_extractor.process_documents([test_doc])
        
        # Verify skill extraction results with confidence
        self.assertEqual(len(processed_docs), 1)
        doc_metadata = processed_docs[0].metadata
        self.assertIn("skill_extraction", doc_metadata)
        self.assertIn("validated_skills", doc_metadata["skill_extraction"])
        self.assertIn("skill_confidence", doc_metadata["skill_extraction"])

    def test_vector_store_integration(self):
        """Test the vector store integration."""
        # Initialize embedding service
        embedding_service = EmbeddingService(qdrant_client=self.qdrant_client)
        
        # Create test documents
        from langchain.schema.document import Document
        test_docs = [
            Document(
                page_content="Python programming and machine learning",
                metadata={"source": "test1.txt", "type": "technical"}
            ),
            Document(
                page_content="Leadership and communication skills",
                metadata={"source": "test2.txt", "type": "soft"}
            )
        ]
        
        # Store documents in vector store
        doc_ids = embedding_service.store_documents(test_docs)
        
        # Verify documents were stored
        self.assertEqual(len(doc_ids), 2)
        
        # Test similarity search
        results = embedding_service.search_similar(
            query="Python programming",
            k=1,
            filter_condition={"type": "technical"}
        )
        
        self.assertTrue(len(results) > 0)
        self.assertIn("Python programming", results[0].page_content)

    def test_full_pipeline_with_advanced_features(self):
        """Test the complete document processing pipeline with advanced features."""
        # Process documents through the complete pipeline
        result = self.parser.parse_document(
            file_path=self.text_file,
            output_dir=self.temp_dir
        )
        
        # Verify pipeline results
        self.assertIn("document_count", result)
        self.assertIn("chunk_count", result)
        self.assertIn("processed_documents", result)
        self.assertIn("consolidated_skill_data", result)
        
        # Verify advanced features
        self.assertIn("chunk_evaluation", result)
        self.assertIn("skill_confidence", result["consolidated_skill_data"])
        
        # Verify output files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "document_embeddings.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "skill_extraction_state.json")))
        
        # Verify vector store collection
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        self.assertIn(self.config["embeddings"]["vector_store"]["collection_name"], collection_names)

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary files
        if os.path.exists(self.text_file):
            os.remove(self.text_file)
        if os.path.exists(self.pdf_file):
            os.remove(self.pdf_file)
        
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
        
        # Clean up Qdrant collection
        try:
            self.qdrant_client.delete_collection(
                self.config["embeddings"]["vector_store"]["collection_name"]
            )
        except Exception:
            pass

if __name__ == "__main__":
    unittest.main() 