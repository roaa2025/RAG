# Complete Document Processing Pipeline

This pipeline connects all components of the document processing system from document extraction to storage in a vector database:

1. **Document Text Extraction** - Extract text from various file formats (PDF, DOCX, images)
2. **Text Preprocessing** - Clean and normalize extracted text
3. **Skill Extraction** - Identify and validate skills mentioned in documents
4. **Advanced Chunking** - Split documents into optimized chunks for better retrieval
5. **Embedding Generation** - Create vector embeddings for document chunks
6. **Vector Store Integration** - Store embeddings in Qdrant for semantic search

## Prerequisites

1. Python 3.8+ with pip
2. OpenAI API key (for embeddings and skill extraction)
3. Qdrant running locally or remotely

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Start Qdrant (if using locally):
   ```
   python run_local_qdrant.py
   ```

## Running the Pipeline

### Option 1: Complete Pipeline (from raw documents)

Use the `run_pipeline.py` script to process one or more documents through the entire pipeline:

```bash
python run_pipeline.py path/to/document.pdf --output-dir output
```

### Option 2: Using Existing Parsed Documents

If you already have parsed documents (in `parsed_documents.json`), you can skip the document extraction and parsing steps using the `process_existing_documents.py` script:

```bash
python process_existing_documents.py output/parsed_documents.json
```

This will:
1. Load the existing parsed documents
2. Process the chunks
3. Create embeddings 
4. Store them in Qdrant

This is useful when:
- You've already run document extraction and want to update embeddings
- You want to load the same documents into a different vector store
- You want to experiment with different embedding models without re-processing documents

#### Command-line Options for Existing Documents

- `json_file`: Path to the parsed documents JSON file (required)
- `--output-dir`: Directory to save output files (default: "output")
- `--collection`: Name of the Qdrant collection (default: "document_embeddings")
- `--force`: Force reprocessing even if embeddings exist
- `--verbose`: Enable verbose output

### Command-line Options for Full Pipeline

- `file_paths`: One or more paths to documents to process
- `--output-dir`: Directory to save output files (default: "output")
- `--config`: Path to custom configuration file (optional)
- `--no-qdrant-store`: Skip storing embeddings in Qdrant
- `--collection`: Qdrant collection name (default: "document_embeddings")
- `--verbose`: Enable verbose logging

### Example Usage

```bash
# Process a single document with default settings
python run_pipeline.py test_files/resume.pdf

# Process multiple documents with custom output directory
python run_pipeline.py test_files/*.pdf --output-dir results

# Use a custom configuration file
python run_pipeline.py test_files/report.docx --config examples/advanced_chunking_config.json

# Process documents without storing in Qdrant
python run_pipeline.py test_files/article.pdf --no-qdrant-store

# Process existing parsed documents
python process_existing_documents.py output/parsed_documents.json

# Process existing parsed documents with custom collection name
python process_existing_documents.py output/parsed_documents.json --collection resume_docs
```

## Pipeline Components

### 1. Document Extraction
- Supports various file formats (PDF, DOCX, TXT, images)
- OCR for images and scanned PDFs
- Layout preservation for tables and structured content

### 2. Text Preprocessing
- Cleaning and normalization
- Language detection
- Duplicate removal
- Special character handling

### 3. Skill Extraction
- NER-based skill identification
- LLM-powered validation and categorization
- Confidence scoring for extracted skills
- Consolidation across multiple documents

### 4. Advanced Chunking
- Multi-layer chunking strategies:
  - Hierarchical: Section-based chunking
  - Semantic: Meaning-based chunking
  - Fixed-size: Token-based chunking
- Metadata preservation
- Position tracking

### 5. Embedding Generation
- OpenAI embeddings (text-embedding-3-small)
- Context-aware embedding
- Metadata enrichment

### 6. Vector Store Integration
- Qdrant storage with appropriate collection setup
- Metadata storage for rich retrieval
- Optimized batch insertion

## Output Files

The pipeline generates several output files:

- `parsed_documents.json`: Raw parsed documents with metadata
- `document_embeddings.json`: Embeddings for document chunks
- `skill_extraction_state.json`: Consolidated skill data
- `skill_extraction_report.md`: Human-readable skill report
- `pipeline_results.json`: Complete pipeline results and metrics

When using the existing documents processor:
- `embedding_results.json`: Results from processing existing documents

## Searching Stored Documents

After processing documents, you can search them using:

```bash
# Basic search
python memory_search.py --query "your search query"

# Advanced search with filters
python search_documents.py --query "your search query" --filter "metadata.source=resume"
```

## Advanced Configuration

The pipeline can be customized using JSON configuration files. See `examples/advanced_chunking_config.json` for chunking options.

Key configuration sections:
- `ocr`: OCR extraction settings
- `preprocessing`: Text preprocessing options
- `skill_extraction`: Skill extraction parameters
- `structuring`: Document chunking strategies
- `embeddings`: Embedding generation options

## Programmatic Usage

You can also use the pipeline programmatically:

```python
# Full pipeline
from app.main import run_complete_pipeline

results = run_complete_pipeline(
    file_paths=["document1.pdf", "document2.docx"],
    output_dir="output",
    config_path="custom_config.json",
    store_in_qdrant=True,
    qdrant_collection="my_documents"
)

# Or process existing documents
from process_existing_documents import process_parsed_documents

results = process_parsed_documents(
    input_json_path="output/parsed_documents.json",
    output_dir="output",
    collection_name="my_documents"
)

# Access results
print(f"Processed {results['document_count']} documents")
print(f"Created {results['embedding_count']} embeddings")
``` 