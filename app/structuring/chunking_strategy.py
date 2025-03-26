from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import re
from collections import defaultdict
import numpy as np
from langchain.schema.document import Document
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Custom sentence tokenizer as fallback for NLTK's sent_tokenize
def custom_sentence_tokenize(text: str) -> List[str]:
    """
    Custom sentence tokenizer as fallback for NLTK's sent_tokenize.
    Uses regex patterns to split text into sentences.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of sentences
    """
    # Simple pattern for sentence boundaries (period, question mark, exclamation mark followed by space and capital letter)
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    
    # Further split very long sentences at common delimiters
    result = []
    for sentence in sentences:
        # If sentence is very long, try to split it further
        if len(sentence) > 300:
            # Split at semicolons, colons, commas, dashes, and similar boundaries
            sub_parts = re.split(r'(?<=[;:])\s+', sentence)
            result.extend(sub_parts)
        else:
            result.append(sentence)
    
    # Filter out empty sentences
    return [s for s in result if s.strip()]

# Safe sentence tokenization function
def safe_sent_tokenize(text: str) -> List[str]:
    """Safely tokenize text into sentences, handling potential errors."""
    try:
        return sent_tokenize(text)
    except Exception:
        # Fallback to simple splitting on periods if NLTK fails
        return [s.strip() + '.' for s in text.split('.') if s.strip()]


class MultiLayerChunker:
    """
    Implementation of a multi-layer chunking strategy for documents with three layers:
    1. Hierarchical Chunking: Split based on document structure
    2. Semantic Chunking: Split based on topic/content similarity
    3. Fixed-Size Chunking: Split based on token limits
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the multi-layer chunker with optional configuration.
        
        Args:
            config: Configuration dictionary for chunking parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "hierarchical": {
                "enabled": True,
                "min_section_length": 100,  # Minimum character length for sections
                "headings_regexp": r'^#+\s+.+|^.+\n[=\-]+$'  # Markdown heading pattern
            },
            "semantic": {
                "enabled": True,
                "similarity_threshold": 0.75,  # Threshold for determining chunks
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "min_sentences_per_chunk": 3,  # Minimum sentences for a chunk
                "max_sentences_to_consider": 20  # Maximum sentences to analyze at once
            },
            "fixed_size": {
                "enabled": True,
                "max_tokens": 1000,  # Maximum tokens per chunk for most LLMs
                "max_chars": 4000,  # Approximate character equivalent
                "overlap_tokens": 100  # Overlap between chunks
            },
            "metadata": {
                "preserve_headers": True,  # Keep headers in chunk metadata
                "include_position": True,  # Include position info in metadata
                "include_chunk_type": True  # Include which chunking method was used
            }
        }
        
        # Merge provided config with defaults
        self.merged_config = self._merge_configs(self.default_config, self.config)
        
        # Initialize embedding model if semantic chunking is enabled
        self.embedding_model = None
        self.tokenizer = None
        
        if self.merged_config["semantic"]["enabled"]:
            self._initialize_embedding_model()
            
        self.logger.info("Initialized MultiLayerChunker with merged config")
    
    def _merge_configs(self, default_config: Dict, user_config: Dict) -> Dict:
        """
        Recursively merge user config with default config.
        
        Args:
            default_config: Default configuration dictionary
            user_config: User-provided configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        merged = default_config.copy()
        
        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
        
    def _initialize_embedding_model(self):
        """Initialize the embedding model for semantic chunking"""
        try:
            model_name = self.merged_config["semantic"]["embedding_model"]
            self.logger.info(f"Loading embedding model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)
            
            self.logger.info(f"Successfully loaded embedding model")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            self.merged_config["semantic"]["enabled"] = False
            self.logger.warning("Disabling semantic chunking due to model loading failure")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embeddings for text using the loaded model.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not self.embedding_model or not self.tokenizer:
            return np.zeros((768,))  # Return zeros if model not loaded
        
        try:
            # Ensure text is not empty and truncate if too long
            if not text or not text.strip():
                return np.zeros((768,))
                
            # Truncate text if it's extremely long to avoid CUDA memory issues and model sequence length limits
            max_chars = 500  # Use a conservative limit well below the model's max sequence length
            if len(text) > max_chars:
                self.logger.warning(f"Truncating text from {len(text)} to {max_chars} characters for embedding")
                text = text[:max_chars]
                
            # Tokenize and get model outputs
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                
            # Mean pooling - take average of all token embeddings
            token_embeddings = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask, 1)
            sum_mask = torch.sum(mask, 1)
            
            # Handle padding
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            # Get mean embeddings
            mean_embeddings = sum_embeddings / sum_mask
            
            return mean_embeddings.squeeze().numpy()
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros((768,))  # Return zeros on error
    
    def _extract_headers_and_sections(self, document: Document) -> List[Dict[str, Any]]:
        """
        Extract headers and content sections from a document based on its structure.
        
        Args:
            document: Document to process
            
        Returns:
            List of dictionaries with header and content information
        """
        sections = []
        content = document.page_content
        header_pattern = self.merged_config["hierarchical"]["headings_regexp"]
        min_section_length = self.merged_config["hierarchical"]["min_section_length"]
        
        # For markdown and structured documents, look for headers
        header_matches = list(re.finditer(header_pattern, content, re.MULTILINE))
        
        # Check if we have Docling metadata with structural information
        if document.metadata.get("dl_meta"):
            dl_meta = document.metadata.get("dl_meta", {})
            
            # If Docling extracted elements, use them for better chunking
            if "elements" in dl_meta and isinstance(dl_meta["elements"], list):
                section_start = 0
                current_header = None
                current_section = []
                
                for element in dl_meta["elements"]:
                    element_type = element.get("type")
                    element_text = element.get("text", "")
                    
                    # Identify headers from Docling elements
                    if element_type in ["heading", "title", "subtitle"]:
                        # If we have a section in progress, save it
                        if current_section and len("".join(current_section)) >= min_section_length:
                            sections.append({
                                "header": current_header or "Untitled Section",
                                "content": "\n".join(current_section),
                                "level": 1 if element_type == "title" else 2,
                                "start_pos": section_start,
                                "end_pos": section_start + len("".join(current_section))
                            })
                        
                        # Start new section
                        current_header = element_text
                        current_section = []
                        section_start = content.find(element_text)
                    else:
                        # Add content to current section
                        current_section.append(element_text)
                
                # Add final section
                if current_section and len("".join(current_section)) >= min_section_length:
                    sections.append({
                        "header": current_header or "Untitled Section",
                        "content": "\n".join(current_section),
                        "level": 2,
                        "start_pos": section_start,
                        "end_pos": section_start + len("".join(current_section))
                    })
                    
                if sections:
                    return sections
        
        # Fallback to regex-based header detection if no Docling elements or sections found
        if header_matches:
            # Process each section (from one header to the next)
            for i, match in enumerate(header_matches):
                header = match.group(0).strip()
                header_start = match.start()
                
                # Check for header level (number of # for markdown)
                level = 1
                if header.startswith('#'):
                    level = len(re.match(r'^#+', header).group(0))
                    header = re.sub(r'^#+\s+', '', header)
                    
                # Get section content (from end of current header to start of next, or end of document)
                if i < len(header_matches) - 1:
                    section_end = header_matches[i + 1].start()
                else:
                    section_end = len(content)
                    
                section_content = content[header_start + len(match.group(0)):section_end].strip()
                
                # Only include sections that have meaningful content
                if len(section_content) >= min_section_length:
                    sections.append({
                        "header": header,
                        "content": section_content,
                        "level": level,
                        "start_pos": header_start,
                        "end_pos": section_end
                    })
        
        # If no headers found, treat the whole document as one section
        if not sections:
            sections.append({
                "header": "Document",
                "content": content,
                "level": 0,
                "start_pos": 0,
                "end_pos": len(content)
            })
            
        return sections
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Chunk text based on semantic similarity.
        
        Args:
            text: Text to chunk semantically
            
        Returns:
            List of semantically coherent text chunks
        """
        if not self.merged_config["semantic"]["enabled"] or not self.embedding_model:
            # Return the whole text as one chunk if semantic chunking is disabled
            return [text]
            
        # Parameters
        similarity_threshold = self.merged_config["semantic"]["similarity_threshold"]
        min_sentences = self.merged_config["semantic"]["min_sentences_per_chunk"]
        max_sentences = self.merged_config["semantic"]["max_sentences_to_consider"]
        
        try:
            # Split text into sentences using safe tokenization
            sentences = safe_sent_tokenize(text)
            
            if len(sentences) <= min_sentences:
                # Not enough sentences to split meaningfully
                return [text]
                
            # Initialize chunks
            chunks = []
            current_chunk = []
            current_embedding = None
            
            # Process sentences in sliding windows of max_sentences
            for i in range(0, len(sentences), max_sentences):
                window = sentences[i:i+max_sentences]
                
                # Get embeddings for each sentence in the window
                embeddings = [self._get_embedding(sent) for sent in window]
                
                for j, sentence in enumerate(window):
                    # If starting a new chunk
                    if not current_chunk:
                        current_chunk.append(sentence)
                        current_embedding = embeddings[j]
                        continue
                    
                    # Calculate similarity with current chunk
                    similarity = cosine_similarity([current_embedding], [embeddings[j]])[0][0]
                    
                    if similarity >= similarity_threshold:
                        # Add to current chunk if similar enough
                        current_chunk.append(sentence)
                        # Update chunk embedding with rolling average
                        weight = len(current_chunk) - 1
                        current_embedding = (current_embedding * weight + embeddings[j]) / (weight + 1)
                    else:
                        # If we have enough sentences, finalize the chunk
                        if len(current_chunk) >= min_sentences:
                            chunks.append(" ".join(current_chunk))
                        
                        # Start a new chunk
                        current_chunk = [sentence]
                        current_embedding = embeddings[j]
            
            # Add the last chunk if it has enough sentences
            if current_chunk and len(current_chunk) >= min_sentences:
                chunks.append(" ".join(current_chunk))
            elif current_chunk:
                # If last chunk is too small, append to previous chunk or create a new one
                if chunks:
                    chunks[-1] = chunks[-1] + " " + " ".join(current_chunk)
                else:
                    chunks.append(" ".join(current_chunk))
                    
            return chunks
        
        except Exception as e:
            self.logger.error(f"Error in semantic chunking: {str(e)}")
            # Return the whole text as one chunk on error
            return [text]
    
    def _token_count(self, text: str) -> int:
        """
        Count the approximate number of tokens in text.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Approximate token count
        """
        try:
            if self.tokenizer:
                # Use actual tokenizer if available
                return len(self.tokenizer.encode(text))
            else:
                # Approximate 4 characters per token as fallback
                return len(text) // 4
        except Exception as e:
            self.logger.error(f"Error counting tokens: {str(e)}")
            # Fallback to character approximation
            return len(text) // 4
    
    def _fixed_size_chunk(self, text: str) -> List[str]:
        """
        Chunk text into fixed-size chunks based on token limits.
        
        Args:
            text: Text to chunk by fixed size
            
        Returns:
            List of fixed-size text chunks
        """
        if not self.merged_config["fixed_size"]["enabled"]:
            # Return the whole text as one chunk if fixed chunking is disabled
            return [text]
            
        max_tokens = self.merged_config["fixed_size"]["max_tokens"]
        overlap_tokens = self.merged_config["fixed_size"]["overlap_tokens"]
        
        try:
            # If text is under the token limit, return as is
            if self._token_count(text) <= max_tokens:
                return [text]
                
            # Split into sentences using safe tokenization
            sentences = safe_sent_tokenize(text)
            chunks = []
            current_chunk = []
            current_token_count = 0
            
            for sentence in sentences:
                sentence_token_count = self._token_count(sentence)
                
                # If this sentence would exceed the limit, finalize the chunk
                if current_token_count + sentence_token_count > max_tokens and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Start a new chunk with overlap
                    overlap_sentences = []
                    overlap_token_count = 0
                    
                    # Add sentences from the end of the previous chunk for overlap
                    for prev_sentence in reversed(current_chunk):
                        if overlap_token_count + self._token_count(prev_sentence) <= overlap_tokens:
                            overlap_sentences.insert(0, prev_sentence)
                            overlap_token_count += self._token_count(prev_sentence)
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_token_count = overlap_token_count
                
                # Add the current sentence to the chunk
                current_chunk.append(sentence)
                current_token_count += sentence_token_count
            
            # Add the final chunk if not empty
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
            return chunks
        
        except Exception as e:
            self.logger.error(f"Error in fixed-size chunking: {str(e)}")
            # If chunking fails, return the whole text
            return [text]
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Apply the multi-layer chunking strategy to a document.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunked Documents
        """
        chunked_docs = []
        
        try:
            # Step 1: Hierarchical Chunking - Split by document structure
            if self.merged_config["hierarchical"]["enabled"]:
                sections = self._extract_headers_and_sections(document)
                
                for section in sections:
                    # Prepare base metadata
                    section_metadata = document.metadata.copy()
                    
                    # Add section-specific metadata
                    if self.merged_config["metadata"]["preserve_headers"]:
                        section_metadata["section_header"] = section["header"]
                        section_metadata["section_level"] = section["level"]
                    
                    if self.merged_config["metadata"]["include_position"]:
                        section_metadata["section_start_pos"] = section["start_pos"]
                        section_metadata["section_end_pos"] = section["end_pos"]
                    
                    if self.merged_config["metadata"]["include_chunk_type"]:
                        section_metadata["chunk_type"] = "hierarchical"
                        section_metadata["chunk_pipeline"] = ["hierarchical"]  # Initialize pipeline for hierarchical chunks
                        section_metadata["layer_info"] = {
                            "hierarchical": {
                                "index": 0,
                                "total": len(sections),
                                "header": section["header"],
                                "level": section["level"]
                            }
                        }
                    
                    # Step 2: Semantic Chunking - Further split sections by semantic coherence
                    if self.merged_config["semantic"]["enabled"]:
                        semantic_chunks = self._semantic_chunk(section["content"])
                        
                        for i, semantic_chunk in enumerate(semantic_chunks):
                            # Update metadata for semantic chunk
                            chunk_metadata = section_metadata.copy()
                            
                            if self.merged_config["metadata"]["include_chunk_type"]:
                                chunk_metadata["chunk_type"] = "semantic"
                                chunk_metadata["chunk_pipeline"] = section_metadata.get("chunk_pipeline", []) + ["semantic"]
                                chunk_metadata["layer_info"] = section_metadata.get("layer_info", {}).copy()
                                chunk_metadata["layer_info"]["semantic"] = {
                                    "index": i,
                                    "total": len(semantic_chunks)
                                }
                            
                            # Step 3: Fixed-Size Chunking - Ensure chunks are within token limits
                            if self.merged_config["fixed_size"]["enabled"]:
                                fixed_chunks = self._fixed_size_chunk(semantic_chunk)
                                
                                for j, fixed_chunk in enumerate(fixed_chunks):
                                    # Update metadata for fixed-size chunk
                                    fixed_metadata = chunk_metadata.copy()
                                    
                                    if self.merged_config["metadata"]["include_chunk_type"]:
                                        fixed_metadata["chunk_type"] = "fixed"
                                        fixed_metadata["chunk_pipeline"] = chunk_metadata["chunk_pipeline"] + ["fixed"]
                                        fixed_metadata["layer_info"] = chunk_metadata["layer_info"].copy()
                                        fixed_metadata["layer_info"]["fixed"] = {
                                            "index": j,
                                            "total": len(fixed_chunks)
                                        }
                                    
                                    # Create new Document for fixed-size chunk
                                    chunked_docs.append(Document(
                                        page_content=fixed_chunk,
                                        metadata=fixed_metadata
                                    ))
                            else:
                                # Skip fixed-size chunking if disabled
                                chunked_docs.append(Document(
                                    page_content=semantic_chunk,
                                    metadata=chunk_metadata
                                ))
                    else:
                        # Skip semantic chunking if disabled, apply fixed-size chunking directly
                        fixed_chunks = self._fixed_size_chunk(section["content"])
                        
                        for j, fixed_chunk in enumerate(fixed_chunks):
                            # Update metadata for fixed-size chunk
                            fixed_metadata = section_metadata.copy()
                            
                            if self.merged_config["metadata"]["include_chunk_type"]:
                                fixed_metadata["chunk_type"] = "fixed"
                                fixed_metadata["chunk_pipeline"] = section_metadata.get("chunk_pipeline", []) + ["fixed"]
                                fixed_metadata["layer_info"] = section_metadata.get("layer_info", {}).copy()
                                fixed_metadata["layer_info"]["fixed"] = {
                                    "index": j,
                                    "total": len(fixed_chunks)
                                }
                            
                            # Create new Document for fixed-size chunk
                            chunked_docs.append(Document(
                                page_content=fixed_chunk,
                                metadata=fixed_metadata
                            ))
            else:
                # Skip hierarchical chunking if disabled, apply semantic chunking directly to whole document
                semantic_chunks = self._semantic_chunk(document.page_content)
                
                for i, semantic_chunk in enumerate(semantic_chunks):
                    # Update metadata for semantic chunk
                    chunk_metadata = document.metadata.copy()
                    
                    if self.merged_config["metadata"]["include_chunk_type"]:
                        chunk_metadata["chunk_type"] = "semantic"
                        chunk_metadata["chunk_pipeline"] = ["semantic"]
                        chunk_metadata["layer_info"] = {
                            "semantic": {
                                "index": i,
                                "total": len(semantic_chunks)
                            }
                        }
                    
                    # Apply fixed-size chunking to semantic chunks
                    fixed_chunks = self._fixed_size_chunk(semantic_chunk)
                    
                    for j, fixed_chunk in enumerate(fixed_chunks):
                        # Update metadata for fixed-size chunk
                        fixed_metadata = chunk_metadata.copy()
                        
                        if self.merged_config["metadata"]["include_chunk_type"]:
                            fixed_metadata["chunk_type"] = "fixed"
                            fixed_metadata["chunk_pipeline"] = chunk_metadata["chunk_pipeline"] + ["fixed"]
                            fixed_metadata["layer_info"] = chunk_metadata["layer_info"].copy()
                            fixed_metadata["layer_info"]["fixed"] = {
                                "index": j,
                                "total": len(fixed_chunks)
                            }
                        
                        # Create new Document for fixed-size chunk
                        chunked_docs.append(Document(
                            page_content=fixed_chunk,
                            metadata=fixed_metadata
                        ))
            
            # If no chunks were created (due to errors), return the original document as a single chunk
            if not chunked_docs:
                self.logger.warning("No chunks created, returning original document as single chunk")
                doc_metadata = document.metadata.copy()
                doc_metadata["chunk_type"] = "original"
                chunked_docs.append(Document(
                    page_content=document.page_content,
                    metadata=doc_metadata
                ))
                
            return chunked_docs
            
        except Exception as e:
            self.logger.error(f"Error in document chunking: {str(e)}")
            # Return the original document as one chunk on error
            doc_metadata = document.metadata.copy()
            doc_metadata["chunk_type"] = "original"
            doc_metadata["chunking_error"] = str(e)
            return [Document(
                page_content=document.page_content,
                metadata=doc_metadata
            )]
    
    def evaluate_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Evaluate the quality of the chunking strategy.
        
        Args:
            chunks: List of chunked Documents
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not chunks:
            return {"error": "No chunks to evaluate"}
            
        try:
            # Calculate metrics
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            token_counts = [self._token_count(chunk.page_content) for chunk in chunks]
            
            # Calculate semantic coherence if semantic chunking is enabled
            semantic_scores = []
            if self.merged_config["semantic"]["enabled"] and self.embedding_model:
                for chunk in chunks:
                    if "semantic" in chunk.metadata.get("layer_info", {}):
                        score = self._calculate_semantic_coherence(chunk.page_content)
                        semantic_scores.append(score)
            
            # Track chunk types and pipelines
            chunk_types = []
            chunk_pipelines = []
            layer_stats = {
                "hierarchical": {"count": 0, "sections": set()},
                "semantic": {"count": 0, "groups": set()},
                "fixed": {"count": 0, "groups": set()}
            }
            
            for chunk in chunks:
                # Track chunk type
                chunk_type = chunk.metadata.get("chunk_type", "unknown")
                chunk_types.append(chunk_type)
                
                # Track pipeline
                pipeline = chunk.metadata.get("chunk_pipeline", [])
                chunk_pipelines.append("->".join(pipeline))
                
                # Track layer statistics
                layer_info = chunk.metadata.get("layer_info", {})
                if "hierarchical" in layer_info:
                    layer_stats["hierarchical"]["count"] += 1
                    layer_stats["hierarchical"]["sections"].add(layer_info["hierarchical"].get("header", "unknown"))
                if "semantic" in layer_info:
                    layer_stats["semantic"]["count"] += 1
                    layer_stats["semantic"]["groups"].add(f"group_{layer_info['semantic'].get('index', 0)}")
                if "fixed" in layer_info:
                    layer_stats["fixed"]["count"] += 1
                    layer_stats["fixed"]["groups"].add(f"group_{layer_info['fixed'].get('index', 0)}")
            
            # Count occurrences
            type_counts = defaultdict(int)
            pipeline_counts = defaultdict(int)
            for chunk_type in chunk_types:
                type_counts[chunk_type] += 1
            for pipeline in chunk_pipelines:
                pipeline_counts[pipeline] += 1
                
            # Build evaluation results
            evaluation = {
                "chunk_count": len(chunks),
                "chunk_sizes": {
                    "min": min(chunk_sizes),
                    "max": max(chunk_sizes),
                    "avg": sum(chunk_sizes) / len(chunk_sizes),
                    "std": np.std(chunk_sizes)
                },
                "token_counts": {
                    "min": min(token_counts),
                    "max": max(token_counts),
                    "avg": sum(token_counts) / len(token_counts),
                    "std": np.std(token_counts)
                },
                "chunk_types": dict(type_counts),
                "chunk_pipelines": dict(pipeline_counts),
                "layer_statistics": {
                    "hierarchical": {
                        "total_chunks": layer_stats["hierarchical"]["count"],
                        "unique_sections": len(layer_stats["hierarchical"]["sections"])
                    },
                    "semantic": {
                        "total_chunks": layer_stats["semantic"]["count"],
                        "unique_groups": len(layer_stats["semantic"]["groups"])
                    },
                    "fixed": {
                        "total_chunks": layer_stats["fixed"]["count"],
                        "unique_groups": len(layer_stats["fixed"]["groups"])
                    }
                }
            }
            
            # Add semantic coherence scores if available
            if semantic_scores:
                evaluation["semantic_coherence"] = {
                    "min": min(semantic_scores),
                    "max": max(semantic_scores),
                    "avg": sum(semantic_scores) / len(semantic_scores),
                    "std": np.std(semantic_scores)
                }
                
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating chunks: {str(e)}")
            return {
                "error": f"Error evaluating chunks: {str(e)}",
                "chunk_count": len(chunks)
            }
    
    def _calculate_semantic_coherence(self, text: str) -> float:
        """
        Calculate semantic coherence score for a chunk of text.
        Higher scores indicate better semantic coherence.
        
        Args:
            text: Text chunk to evaluate
            
        Returns:
            Float between 0 and 1 indicating semantic coherence
        """
        try:
            # Split text into sentences
            sentences = safe_sent_tokenize(text)
            
            if len(sentences) < 2:
                return 1.0  # Single sentence is perfectly coherent
                
            # Get embeddings for all sentences
            embeddings = [self._get_embedding(sent) for sent in sentences]
            
            # Calculate pairwise similarities between all sentences
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(similarity)
            
            if not similarities:
                return 1.0  # No pairs to compare
                
            # Return average similarity as coherence score
            return sum(similarities) / len(similarities)
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic coherence: {str(e)}")
            return 0.0  # Return 0 on error to indicate no coherence 