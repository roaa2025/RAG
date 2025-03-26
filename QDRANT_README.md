# Qdrant Vector Database Setup

This guide explains how to set up and use Qdrant as a vector database for storing and retrieving document embeddings in this project.

## What is Qdrant?

Qdrant is a vector similarity search engine designed for building AI applications. It provides a fast and efficient way to store vector embeddings and search for similar vectors, making it ideal for applications like semantic search, recommendation systems, and more.

## Setup Options

There are four ways to set up Qdrant:

### 1. In-Memory Mode (for Testing)

This option runs Qdrant entirely in memory, making it perfect for testing and development. Data will be lost when the process ends.

```python
from qdrant_client import QdrantClient

# Create an in-memory client
client = QdrantClient(":memory:")
```

### 2. Docker Mode (for Persistence)

For a more persistent solution, you can run Qdrant in Docker:

```bash
# Pull and run the Qdrant Docker image
docker run -p 6333:6333 qdrant/qdrant
```

Then connect to it from your Python code:

```python
from qdrant_client import QdrantClient

# Connect to Docker instance
client = QdrantClient("localhost", port=6333)
```

### 3. Docker Compose (Recommended for Production)

For a complete orchestrated setup with both Qdrant and your application, use Docker Compose:

```bash
# Start the entire stack
docker-compose up

# Or run in detached mode
docker-compose up -d
```

This will start both the Qdrant service and your Python application as defined in the `docker-compose.yml` file.

The Python application can connect to Qdrant using the service name as hostname:

```python
from qdrant_client import QdrantClient

# In Docker Compose environment
client = QdrantClient(host="qdrant", port=6333)
```

### 4. Local Binary Installation (without Docker)

If Docker isn't available, you can download and run Qdrant executable directly:

1. Download the appropriate binary from [Qdrant releases page](https://github.com/qdrant/qdrant/releases)
2. Extract the archive
3. Run the executable:

```bash
# Linux/macOS
./qdrant

# Windows
qdrant.exe
```

You can specify a config file or storage path:

```bash
# With config file
qdrant.exe --config-path qdrant_config.yaml

# With storage path
qdrant.exe --storage-path ./qdrant_storage
```

Connect to the local server in your code:

```python
from qdrant_client import QdrantClient

# Connect to local Qdrant instance
client = QdrantClient(url="http://localhost:6333")
```

## Installation

1. Install the Qdrant client:

```bash
pip install qdrant-client
```

2. This is already included in the project's `requirements.txt`, so you can also run:

```bash
pip install -r requirements.txt
```

## Using Docker Compose (Recommended)

This project includes Docker Compose configuration for easy deployment:

1. Make sure Docker and Docker Compose are installed.

2. Start the services:
   ```bash
   docker-compose up
   ```

3. To rebuild the application after changes:
   ```bash
   docker-compose build
   docker-compose up
   ```

4. To run in the background:
   ```bash
   docker-compose up -d
   ```

5. To stop the services:
   ```bash
   docker-compose down
   ```

6. To view logs:
   ```bash
   docker-compose logs -f
   ```

## Provided Scripts

This project includes several scripts for working with Qdrant:

### 1. `setup_qdrant.py`

This script demonstrates how to set up Qdrant in both in-memory mode and Docker mode. It includes tests to verify connectivity and basic functionality.

```bash
python setup_qdrant.py
```

### 2. `qdrant_example.py`

This script provides a more practical example of using Qdrant with text embeddings for document retrieval. It includes a `QdrantDocumentStore` class that wraps around Qdrant and provides a simple interface for adding and searching documents.

```bash
python qdrant_example.py
```

### 3. `local_qdrant_example.py`

This script demonstrates how to connect to a locally running Qdrant instance (either via Docker or binary installation).

```bash
python local_qdrant_example.py
```

### 4. `download_qdrant.py`

This script helps download the Qdrant binary for your operating system (useful if you don't want to use Docker).

```bash
python download_qdrant.py
```

## Local Qdrant Configuration

You can customize your local Qdrant instance using a configuration file. We've provided an example in `qdrant_config.yaml`:

```yaml
storage:
  # Storage persistence path
  storage_path: ./qdrant_storage
  # Use on-disk storage, instead of in-memory
  on_disk_payload: true
  # Enable optimizers in storage
  optimizers_enable: true

service:
  # Port to listen on
  http_port: 6333
  # Address to bind the service to
  host: 127.0.0.1
```

## Using Qdrant in Your Code

### Basic Usage

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Connect to Qdrant (choose one option)
client = QdrantClient(":memory:")  # In-memory
# OR
client = QdrantClient("localhost", port=6333)  # Docker or local binary
# OR
client = QdrantClient(url="http://localhost:6333")  # Alternative syntax
# OR (in Docker Compose)
client = QdrantClient(host="qdrant", port=6333)  # Use service name as host

# Create a collection
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# Add vectors
from qdrant_client.http import models
client.upsert(
    collection_name="my_collection",
    points=models.Batch(
        ids=[1, 2, 3],
        vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]],
        payloads=[{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
    )
)

# Search for similar vectors
search_results = client.search(
    collection_name="my_collection",
    query_vector=[0.2, 0.3, ...],
    limit=5
)
```

### Using the Included QdrantDocumentStore

The `QdrantDocumentStore` class in `qdrant_example.py` provides a higher-level interface for working with documents:

```python
from qdrant_example import QdrantDocumentStore

# Create a document store
doc_store = QdrantDocumentStore(
    use_in_memory=True,  # Set to False for Docker or local binary
    embedding_model="all-MiniLM-L6-v2",  # Choose your embedding model
    host="qdrant"  # When using Docker Compose
)

# Add documents
documents = ["Document 1 text", "Document 2 text", "Document 3 text"]
metadatas = [{"source": "web"}, {"source": "pdf"}, {"source": "book"}]
doc_store.add_documents(documents, metadatas)

# Search for similar documents
results = doc_store.search("your search query", limit=3)
```

## Further Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client API Reference](https://qdrant.github.io/qdrant_client/)
- [Qdrant GitHub Repository](https://github.com/qdrant/qdrant)
- [Qdrant Releases](https://github.com/qdrant/qdrant/releases)
- [Docker Compose Documentation](https://docs.docker.com/compose/) 