import json
import os

config = {
    "structuring": {
        "use_advanced_chunking": True,
        "chunking": {
            "hierarchical": {
                "enabled": True,
                "min_section_length": 100,
                "headings_regexp": "^#+\\s+.+|^.+\\n[=\\-]+$"
            },
            "semantic": {
                "enabled": True,
                "similarity_threshold": 0.75,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "min_sentences_per_chunk": 3,
                "max_sentences_to_consider": 20
            },
            "fixed_size": {
                "enabled": True,
                "max_tokens": 1000,
                "overlap_tokens": 100
            },
            "metadata": {
                "preserve_headers": True,
                "include_position": True,
                "include_chunk_type": True,
                "track_pipeline": True,
                "show_all_layers": True
            }
        }
    }
}

os.makedirs("examples", exist_ok=True)
with open("examples/advanced_chunking_config.json", "w") as f:
    json.dump(config, f, indent=4) 