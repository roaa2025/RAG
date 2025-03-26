# Qdrant Vector Search with LangChain Integration

This project provides tools to store document embeddings in Qdrant and query them using LangChain's similarity search capabilities.

## Overview

The workflow consists of three main parts:

1. **Creating the Qdrant Collection**: Set up a Qdrant collection with dynamic vector size
2. **Loading Embeddings**: Load pre-generated embeddings from JSON and store them in Qdrant
3. **Semantic Search**: Query the Qdrant collection using LangChain's similarity search

## Prerequisites

- Python 3.8+
- Qdrant server running locally or remotely (optional - you can use in-memory mode)
- OpenAI API key (set in your `.env` file)

## Installation

1. Install the required dependencies:

```bash
pip install langchain langchain-openai langchain-community qdrant-client tqdm python-dotenv
```

2. Make sure your OpenAI API key is set in your `.env` file:

```
OPENAI_API_KEY=your-api-key-here
```

## Quick Start: In-Memory Mode (No Qdrant Server Required)

If you don't have a Qdrant server running, you can use in-memory mode for testing:

```bash
# Set up an in-memory Qdrant collection with sample data
python setup_memory_qdrant.py

# Search the in-memory collection
python search_documents.py "vector database" --host=":memory:"
```

This is perfect for development, testing, or when you're unable to start a Qdrant server.

## Standard Workflow

If you have a Qdrant server running, you can follow the full workflow:

### Step 1: Create Qdrant Collection

First, create a Qdrant collection with the appropriate vector size for your embedding model:

```bash
python create_document_embeddings_collection.py --mode=local --model=text-embedding-3-small
```

Options:
- `--mode`: Connection mode (`memory`, `local`, or `remote`)
- `--host`: Qdrant host (for remote mode)
- `--port`: Qdrant port (default: 6333)
- `--model`: Embedding model to determine vector size

### Step 2: Load Embeddings into Qdrant

Load your pre-generated embeddings from JSON and store them in the Qdrant collection:

```bash
python load_embeddings_to_qdrant.py --file=output/document_embeddings.json --batch_size=100
```

Options:
- `--file`: Path to the JSON file with embeddings
- `--collection`: Name of the Qdrant collection (default: document_embeddings)
- `--batch_size`: Number of embeddings to insert in a single batch
- `--host`: Qdrant host (default: localhost)
- `--port`: Qdrant port (default: 6333)

### Step 3: Perform Semantic Searches

Query the Qdrant collection using semantic search:

```bash
python search_documents.py "your search query here" --top_k 3 --with_scores
```

Options:
- First argument: Your search query
- `--top_k`: Number of results to return (default: 3)
- `--with_scores`: Include similarity scores in the output
- `--host`: Qdrant host (default: localhost, use ":memory:" for in-memory mode)
- `--port`: Qdrant port (default: 6333)
- `--collection`: Collection name (default: document_embeddings)

## Using the QdrantSearchService in Your Code

You can also use the `QdrantSearchService` class in your own Python code:

```python
from app.qdrant_search import QdrantSearchService

# Initialize the search service (with local Qdrant server)
search_service = QdrantSearchService(
    collection_name="document_embeddings",
    host="localhost",
    port=6333
)

# Or use in-memory mode (no server required)
in_memory_service = QdrantSearchService(
    collection_name="document_embeddings",
    host=":memory:"
)

# Perform a similarity search
results = search_service.similarity_search(
    query="your search query",
    k=3  # number of results to return
)

# Process the results
for doc in results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print()

# Alternatively, get results with similarity scores
results_with_scores = search_service.similarity_search_with_scores(
    query="your search query",
    k=3
)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print()
```

## Using Filters in Queries

You can also filter results using Qdrant's filtering capabilities:

```python
# Filter based on metadata
filter_condition = {
    "must": [
        {
            "key": "metadata.source",
            "match": {
                "value": "your-source-name"
            }
        }
    ]
}

# Search with filter
results = search_service.similarity_search(
    query="your search query",
    k=5,
    filter_condition=filter_condition
)
```

## Troubleshooting

### Connection Issues

If you get a connection error like:
```
[WinError 10061] No connection could be made because the target machine actively refused it
```

You have two options:

1. **Use in-memory mode** (easiest): 
   ```bash
   python setup_memory_qdrant.py
   python search_documents.py "your query" --host=":memory:"
   ```

2. **Setup a local Qdrant server**:
   - If you have Docker: `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`
   - Download the Qdrant binary from https://qdrant.tech/documentation/guides/installation/
   - Follow platform-specific installation instructions

### Other Issues

- If the search returns unexpected results, check that the embeddings were properly loaded into Qdrant
- Make sure your OpenAI API key is valid and set correctly in your `.env` file

## Performance Considerations

- Adjust the batch size when loading embeddings based on your system's memory
- For large collections, consider using Qdrant's quantization options
- The similarity search performance depends on the size of your collection and the complexity of your queries
- In-memory mode works great for development but isn't suitable for production or large datasets

## Additional Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started)
- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings) 