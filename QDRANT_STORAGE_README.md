# Storing and Searching Document Embeddings in Qdrant

This README explains how to store document embeddings (with skills metadata) in Qdrant and how to search them.

## Prerequisites

1. **Qdrant Server**:
   - Must have a Qdrant server running
   - The default connection is to `localhost:6333`
   - You can run Qdrant using Docker: `docker run -p 6333:6333 qdrant/qdrant`

2. **OpenAI API Key**:
   - Store your API key in a `.env` file: `OPENAI_API_KEY=your_api_key_here`
   - This is required for creating embeddings and searching

3. **Python Dependencies**:
   - Make sure you have all dependencies installed: `pip install -r requirements.txt`

## Complete Workflow

### Step 1: Process Documents & Create Embeddings

First, process your documents to extract text, skills, and create embeddings:

```bash
python run_docling.py --create-embeddings
```

This will:
- Process documents with advanced chunking (now the default)
- Extract skills from documents
- Create embeddings for all chunks
- Save embeddings to `output/document_embeddings.json`

### Step 2: Store Embeddings in Qdrant

Use the `store_in_qdrant.py` script to store the embeddings in Qdrant:

```bash
python store_in_qdrant.py --verify
```

Optional arguments:
- `--embeddings-file PATH`: Path to embeddings JSON file
- `--collection NAME`: Name of the Qdrant collection
- `--batch-size N`: Number of embeddings to insert in a single batch
- `--verify`: Verify storage with a test query
- `--query TEXT`: Test query for verification

### Step 3: Search for Documents

Use the `search_qdrant.py` script to search for documents:

```bash
python search_qdrant.py "your search query here"
```

Optional arguments:
- `--collection NAME`: Name of the Qdrant collection
- `--results N`: Number of results to return
- `--filter-skill SKILL`: Only return documents with this skill
- `--no-content`: Don't include document content in results
- `--no-skills`: Don't show skills in results
- `--output PATH`: Save results to this file

Examples:
```bash
# Basic search
python search_qdrant.py "data warehousing techniques"

# Search with skill filter
python search_qdrant.py "data analysis" --filter-skill "Data Integration"

# Get more results and save to file
python search_qdrant.py "GIS technology" --results 10 --output search_results.json
```

## Alternative Ways to Store Embeddings

### Option 1: Using the Complete Pipeline

For a complete pipeline that also stores in Qdrant:

```bash
python -c "from app.main import run_complete_pipeline; run_complete_pipeline(['path/to/documents/*.pdf'], store_in_qdrant=True)"
```

### Option 2: Using Python Code

You can also do this in your Python code:

```python
from app.embedding_service import EmbeddingService

# Initialize the embedding service
embedding_service = EmbeddingService()

# Load embeddings from the JSON file
embeddings_path = "output/document_embeddings.json"

# Store in Qdrant
result = embedding_service.load_and_store_embeddings(
    json_file_path=embeddings_path,
    collection_name="document_embeddings"
)

print(f"Stored {result['embeddings_inserted']} embeddings in Qdrant")
```

## Troubleshooting

1. **Qdrant Connection Issues**:
   - Ensure Qdrant is running and accessible at `localhost:6333`
   - Check the connection with: `curl http://localhost:6333/collections`

2. **Missing Skills in Search Results**:
   - Verify that the skills were extracted properly in the preprocessing step
   - Check the JSON file to see if skills are included in the metadata

3. **Embedding Creation Fails**:
   - Ensure your OpenAI API key is valid and has sufficient quota
   - Check for any rate limit errors in the logs

4. **No Search Results**:
   - Try with a more general query
   - Verify that embeddings were successfully stored in Qdrant
   - Check if the collection exists in Qdrant 