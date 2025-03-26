# Qdrant Setup Instructions

This document provides step-by-step instructions for setting up Qdrant as a vector store using either Docker or direct binary installation.

## Option 1: Using Docker (Recommended)

If you have Docker installed, this is the easiest way to get Qdrant running with persistent storage.

### Prerequisites
- Docker installed
- Docker Compose installed (optional, but recommended)

### Single Container Setup
```bash
# Run Qdrant in a Docker container with persistent storage
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Docker Compose Setup
The project includes a Docker Compose configuration for running both Qdrant and your Python application together.

1. Start the services:
```bash
docker-compose up
```

2. Stop the services:
```bash
docker-compose down
```

## Option 2: Using In-Memory Qdrant (for Testing Only)

For quick testing or development, you can use Qdrant's in-memory mode directly from Python without any installation:

```python
from qdrant_client import QdrantClient

# Create an in-memory client
client = QdrantClient(":memory:")
```

Run the example script to test:
```bash
python qdrant_example.py
```

## Option 3: Using Local Binary (No Docker)

If you don't have Docker, you can download and run the Qdrant binary directly.

### Steps:
1. Run our helper script to download the appropriate binary for your system:
```bash
python download_qdrant.py
```

2. This will create a `qdrant_binary` folder with the Qdrant executable and a batch file to run it.

3. Run Qdrant:
   - On Windows: `qdrant_binary\run_qdrant.bat`
   - On Linux/macOS: `./qdrant_binary/qdrant --storage-path ./qdrant_storage`

4. Test the connection:
```bash
python local_qdrant_example.py
```

## Connecting to Qdrant

After setting up Qdrant using any of the methods above, you can connect to it in your code:

```python
from qdrant_client import QdrantClient

# For local binary or Docker on same machine
client = QdrantClient(host="localhost", port=6333)

# OR
client = QdrantClient(url="http://localhost:6333")

# For Docker Compose environment
# client = QdrantClient(host="qdrant", port=6333)  # Use service name as hostname
```

## Troubleshooting

- If you're having trouble with Docker, ensure Docker is installed and running: `docker --version`
- If you're having connection issues with Qdrant, check if it's running: try opening http://localhost:6333/dashboard/ in your browser
- For in-memory mode, no setup is required, just ensure qdrant-client is installed via pip 