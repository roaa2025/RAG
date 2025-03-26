# Skill Extraction Feature

This document explains how to use the skill extraction feature of the Docling application. The skill extraction module uses Natural Language Processing (NLP) and Named Entity Recognition (NER) to extract, validate, and categorize domain-specific skills from unstructured text.

## Overview

The skill extraction feature follows a 3-step process:

1. **Extract Skills Using NER**: Identifies potential skills, technologies, or competencies using Named Entity Recognition (NER).
2. **Validate Skills Using Contextual Analysis**: Verifies whether each extracted skill is relevant and meaningful based on the surrounding text.
3. **Categorize Skills**: Categorizes skills into 'validated' and 'rejected' lists based on relevance.

## Configuration

The skill extraction feature can be configured using the following settings:

```json
{
  "skill_extraction": {
    "enabled": true,
    "openai_api_key": "your-api-key",
    "method": "openai",
    "min_confidence": 0.5,
    "store_rejected": true
  }
}
```

- `enabled`: Set to `true` to enable skill extraction, `false` to disable.
- `openai_api_key`: Your OpenAI API key for accessing GPT models.
- `method`: Currently only "openai" is supported.
- `min_confidence`: Threshold for skill confidence (not currently used).
- `store_rejected`: Whether to store rejected skills in the output.

## Setting up the OpenAI API Key

You can set the OpenAI API key in one of three ways:

1. **Environment Variable**: Set the `OPENAI_API_KEY` environment variable.
   ```bash
   export OPENAI_API_KEY=sk-your-api-key
   ```

2. **Configuration File**: Include it in a configuration file:
   ```json
   {
     "skill_extraction": {
       "enabled": true,
       "openai_api_key": "sk-your-api-key"
     }
   }
   ```

3. **Direct Code**: Pass it directly in the code (not recommended for production):
   ```python
   config = {
       "skill_extraction": {
           "enabled": True,
           "openai_api_key": "sk-your-api-key"
       }
   }
   ```

## Output Format

The skill extraction module outputs data in the following JSON format:

```json
{
  "text": "Original text input",
  "state": {
    "validated_skills": ["Python", "Machine Learning", "SQL"],
    "rejected_skills": ["Data", "Coding"]
  }
}
```

This output is stored in the document metadata under `metadata.dl_meta.skills` and also in `metadata.skill_extraction`.

## Running the Example

To run the skill extraction example:

```bash
python examples/skill_extraction_example.py
```

This will process a sample text and output the extracted skills.

## Usage in Your Code

To use skill extraction in your code:

```python
from app.main import parse_document

# Define configuration with skill extraction enabled
config = {
    "skill_extraction": {
        "enabled": True,
        "openai_api_key": "your-api-key"
    }
}

# Parse a document with skill extraction
result = parse_document("path/to/document.txt", config=config, output_dir="output")

# Access the extracted skills
for doc in result["processed_documents"]:
    skills = doc["metadata"]["skill_extraction"]
    print(f"Validated Skills: {skills['validated_skills']}")
    print(f"Rejected Skills: {skills['rejected_skills']}")
```

## Limitations

- The skill extraction module currently relies on OpenAI's API, which requires an internet connection and API key.
- The quality of extracted skills depends on the quality of the input text and the capabilities of the underlying model.
- Processing large documents may take time and consume API credits.

---

For more information on the Docling application, see the main README.md file. 