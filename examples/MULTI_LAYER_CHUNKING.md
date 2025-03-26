# Multi-Layer Chunking Strategy in Docling

This document explains the multi-layer chunking strategy implemented in Docling for handling large datasets containing PDFs and images.

## Overview

The multi-layer chunking strategy addresses the challenge of processing large documents by dividing them into meaningful, context-preserving chunks. This implementation uses a three-layer approach:

1. **Hierarchical Chunking** (First Layer): Splits documents into sections and subsections based on document structure.
2. **Semantic Chunking** (Second Layer): Further splits sections into semantically coherent chunks based on meaning and topic relevance.
3. **Fixed-Size Chunking** (Third Layer): Applies a final layer of chunking to limit chunk size for LLM compatibility.

## Chunking Layers Explained

### Layer 1: Hierarchical Chunking

This layer identifies the document's natural structure by:
- Detecting headings, titles, and section boundaries
- Preserving the logical organization of the document
- Maintaining the document hierarchy for context

Implemented using:
- Regex pattern matching for markdown-style headers
- Docling's document structure recognition
- Element type classification (heading, title, etc.)

### Layer 2: Semantic Chunking

This layer ensures semantic coherence by:
- Analyzing sentence similarity within sections
- Grouping related content by topic
- Preserving complete thoughts rather than arbitrary breaks

Implemented using:
- Sentence embedding with Sentence Transformers
- Cosine similarity measurement between sentences
- Dynamic threshold-based grouping

### Layer 3: Fixed-Size Chunking

This layer optimizes for LLM token limitations by:
- Ensuring chunks don't exceed maximum token limits
- Adding overlap between chunks to maintain context
- Preserving sentence boundaries for readability

Implemented using:
- Tokenizer-based token counting
- Sentence-level splitting to maintain coherence
- Configurable overlap to preserve context

## Using the Multi-Layer Chunking Strategy

### Command Line Usage

Enable advanced chunking with the `-a` flag:

```bash
python -m app.main document.pdf --output output_dir --advanced-chunking
```

### Python API Usage

```python
from app.main import parse_document

# Enable advanced chunking via configuration
config = {
    "structuring": {
        "use_advanced_chunking": True
    }
}

result = parse_document("document.pdf", config=config, output_dir="output_dir")
```

### Configuration Options

The chunking strategy is highly configurable. Create a configuration file like:

```json
{
    "structuring": {
        "use_advanced_chunking": true,
        "chunking": {
            "hierarchical": {
                "enabled": true,
                "min_section_length": 100,
                "headings_regexp": "^#+\\s+.+|^.+\\n[=\\-]+$"
            },
            "semantic": {
                "enabled": true,
                "similarity_threshold": 0.75,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "min_sentences_per_chunk": 3,
                "max_sentences_to_consider": 20
            },
            "fixed_size": {
                "enabled": true,
                "max_tokens": 1000,
                "overlap_tokens": 100
            }
        }
    }
}
```

## Testing the Chunking Strategy

Run the example test script:

```bash
python examples/test_multi_layer_chunking.py
```

This will:
1. Process documents in the `examples/documents/` directory
2. Apply the multi-layer chunking strategy
3. Output detailed evaluation metrics
4. Save the structured and chunked results to the output directory

## Evaluation Metrics

The implementation includes a comprehensive evaluation system:

- **Token count statistics**: Min, max, average, and std dev of token counts
- **Chunk size statistics**: Distribution of chunk sizes
- **Semantic coherence**: Measures the semantic similarity within chunks
- **Chunk type distribution**: Number of chunks from each layer of the strategy

These metrics help you understand the effectiveness of the chunking strategy and tune parameters for your specific use case.

## Advanced Use Cases

### Tuning for Different Document Types

Different document types benefit from different chunking strategies:

- **Academic papers**: Increase hierarchical chunking sensitivity to detect sections and subsections
- **Legal documents**: Adjust semantic thresholds to group related legal concepts
- **Technical documentation**: Configure fixed-size chunking for optimal API reference handling

### Disabling Specific Layers

You can disable any layer that doesn't fit your use case:

```json
{
    "structuring": {
        "chunking": {
            "hierarchical": { "enabled": true },
            "semantic": { "enabled": false },
            "fixed_size": { "enabled": true }
        }
    }
}
```

### Custom Embedding Models

For specialized domains, you can configure different embedding models:

```json
{
    "structuring": {
        "chunking": {
            "semantic": {
                "embedding_model": "your-specialized-domain-model"
            }
        }
    }
}
``` 