# Document Parsing Application

A comprehensive document parsing application that leverages [Docling](https://github.com/DS4SD/docling) through LangChain integration to extract text from various document formats with high accuracy while preserving document structure.

## Features

- **Text Extraction**: Extract text from scanned images, PDFs, and handwritten documents with high accuracy using Docling's advanced OCR capabilities.
- **Layout Handling**: Process multi-column layouts, tabular data, and both structured and unstructured documents seamlessly.
- **Text Preprocessing**: Implement text cleaning, error correction, and language detection for high-quality output.
- **Multi-Layer Chunking**: Apply hierarchical, semantic, and fixed-size chunking strategies for optimal document segmentation.
- **Clear Separation of Concerns**:
  - OCR Extraction: Extract text from diverse document types
  - Text Preprocessing: Clean and normalize extracted text
  - Data Structuring: Organize processed text into structured formats
  - Advanced Chunking: Intelligently segment documents for LLM processing
- **Skill Extraction**: Identify, validate, and categorize professional skills in text using NLP/NER

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/docling-parser.git
cd docling-parser
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Parse documents using the command-line interface:

```bash
python -m app.main document_path.pdf --output output_dir --config config.json
```

Enable advanced multi-layer chunking with the `--advanced-chunking` flag:

```bash
python -m app.main document_path.pdf --output output_dir --advanced-chunking
```

You can also parse multiple documents at once:

```bash
python -m app.main doc1.pdf doc2.docx doc3.jpg --output output_dir
```

### Python API

Import and use the parser in your Python code:

```python
from app.main import parse_document

# Parse a single document
result = parse_document("path/to/document.pdf", output_dir="output_dir")

# Parse multiple documents
documents = ["doc1.pdf", "doc2.docx", "doc3.jpg"]
result = parse_document(documents, output_dir="output_dir")

# Use custom configuration
custom_config = {
    "ocr": {
        "convert_kwargs": {
            "ocr_lang": "fra",  # French
            "ocr_dpi": 600,     # Higher quality OCR
        }
    }
}
result = parse_document("document.pdf", config=custom_config)

# Enable advanced chunking
chunking_config = {
    "structuring": {
        "use_advanced_chunking": True,
        "chunking": {
            "fixed_size": {
                "max_tokens": 1500,  # Custom token limit
                "overlap_tokens": 150  # Custom overlap
            }
        }
    }
}
result = parse_document("document.pdf", config=chunking_config)
```

## Component Architecture

The application is built with a modular architecture to maintain separation of concerns:

### 1. OCR Extraction (`app/ocr/`)

The OCR component focuses solely on text extraction from documents, using Docling's advanced OCR capabilities.

- **Key Features**: 
  - Support for scanned images, PDFs, and handwritten documents
  - Handling of varied document quality
  - Integration with Docling through LangChain

### 2. Text Preprocessing (`app/preprocessing/`)

This component handles cleaning and normalizing the extracted text, improving its quality for downstream tasks.

- **Key Features**:
  - Text cleaning to remove noise and artifacts
  - OCR error correction
  - Language detection and normalization
  - Text standardization

### 3. Data Structuring (`app/structuring/`)

The structuring component organizes the cleaned text into structured formats based on document layout.

- **Key Features**:
  - Layout type identification (single-column, multi-column, table)
  - Table extraction and structuring
  - Multi-column content handling
  - Document metadata preservation

### 4. Multi-Layer Chunking (`app/structuring/chunking_strategy.py`)

The advanced chunking component implements a three-layer approach to optimize documents for LLM processing.

- **Key Features**:
  - Hierarchical chunking based on document structure
  - Semantic chunking using embeddings and similarity analysis
  - Fixed-size chunking with configurable token limits
  - Chunk quality evaluation metrics
  - Extensive configuration options

### 5. Utilities (`app/utils/`)

Utility functions for document handling, configuration management, and file operations.

- **Key Components**:
  - Document type detection
  - File handling utilities
  - Configuration management
  - JSON export functions

## Configuration

The application supports customization through a configuration file. Here's an example configuration:

```json
{
  "ocr": {
    "convert_kwargs": {
      "use_ocr": true,
      "ocr_lang": "eng",
      "ocr_dpi": 300,
      "force_ocr": false
    }
  },
  "preprocessing": {
    "ocr_corrections": {
      "l1": "h",
      "rn": "m"
    },
    "normalization": {
      "lowercase": true,
      "remove_extra_whitespace": true
    }
  },
  "structuring": {
    "use_advanced_chunking": true,
    "table_extraction": {
      "detect_delimiters": true
    },
    "layout": {
      "detect_columns": true
    },
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
        "min_sentences_per_chunk": 3
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

## Output Format

The parser outputs a structured JSON format that includes:

- Input file information
- Document count and chunk count
- Processed documents, including:
  - Content type (single-column, multi-column, table)
  - Metadata (source, language, etc.)
  - Raw text (original processed text)
  - Structured content (organized by layout type)
  - Chunk information (when using advanced chunking)

When using advanced chunking, the output also includes:
- Chunk evaluation metrics
- Hierarchical structure of chunks
- Semantic coherence scores
- Token count statistics

Example output:
```json
{
  "input_files": ["document.pdf"],
  "document_count": 1,
  "total_chunk_count": 15,
  "chunk_evaluation": {
    "document_0": {
      "chunk_count": 15,
      "chunk_sizes": {
        "min": 120,
        "max": 1250,
        "avg": 850.3,
        "std": 315.2
      },
      "chunk_types": {
        "hierarchical": 5,
        "semantic": 2,
        "fixed": 8
      }
    }
  },
  "documents": [...],
  "chunks": [...]
}
```

## Advanced Chunking

For detailed information about the multi-layer chunking strategy, see [MULTI_LAYER_CHUNKING.md](examples/MULTI_LAYER_CHUNKING.md).

## Skill Extraction (NEW!)

This application now includes a powerful skill extraction feature that uses Natural Language Processing (NLP) and Named Entity Recognition (NER) to:

1. Extract potential skills from text
2. Validate skills using contextual analysis
3. Categorize skills into validated and rejected lists

The skill extraction feature integrates with OpenAI's API for accurate skill identification. 

For detailed documentation, see [README_SKILL_EXTRACTION.md](README_SKILL_EXTRACTION.md).

### Quick Example

```python
from app.main import parse_document

config = {
    "skill_extraction": {
        "enabled": True,
        "openai_api_key": "your-api-key"
    }
}

result = parse_document("resume.pdf", config=config)
```

The extracted skills are stored in the document metadata and accessible via:

```json
{
  "text": "Original text input",
  "state": {
    "validated_skills": ["Python", "Machine Learning", "SQL"],
    "rejected_skills": ["Data", "Coding"]
  }
}
```

## License

MIT

## Credits

This application uses [Docling](https://github.com/DS4SD/docling) and [LangChain](https://github.com/langchain-ai/langchain) for document processing. 