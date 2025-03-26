from typing import List, Union, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, Distance
import time
from tqdm import tqdm
import uuid
from openai import OpenAI

load_dotenv()


class EmbeddingService:
    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        """
        Initialize the embedding service with OpenAI's embedding model.
        
        Args:
            qdrant_client: Optional QdrantClient instance. If not provided,
                          a local client will be initialized.
        """
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize Qdrant client
        if qdrant_client is None:
            # Use persistent client instead of in-memory
            self.qdrant_client = QdrantClient(host="localhost", port=6333)
        else:
            self.qdrant_client = qdrant_client
            
        # Ensure collection exists with proper configuration BEFORE creating vectorstore
        self._ensure_collection_exists()
            
        # Initialize LangChain's Qdrant vectorstore
        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name="document_embeddings",
            embeddings=self.embeddings,
            content_payload_key="page_content",  # Tell Qdrant where to find the text content
            metadata_payload_key="metadata"      # Tell Qdrant where to find the metadata
        )

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists with proper configuration."""
        try:
            # Try to get collection info
            collection_info = self.qdrant_client.get_collection("document_embeddings")
            
            # Check if vector size matches
            if collection_info.config.params.vectors.size != 1536:
                # Delete and recreate with correct size
                self.qdrant_client.delete_collection("document_embeddings")
                raise ValueError("Collection exists with wrong vector size")
                
        except Exception:
            # Get vector size from a sample embedding
            sample_text = "Sample text for vector size determination"
            sample_embedding = self.embeddings.embed_query(sample_text)
            vector_size = len(sample_embedding)
            
            # Create collection with proper configuration
            self.qdrant_client.create_collection(
                collection_name="document_embeddings",
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created 'document_embeddings' collection in Qdrant with vector size {vector_size}")

    def create_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Create embeddings for a single text or list of texts.

        Args:
            texts: Single text string or list of text strings

        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]

        # Create embeddings
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings

    def process_documents(self, texts: Union[str, List[str]]) -> List[Document]:
        """
        Process and split documents into chunks.

        Args:
            texts: Single text string or list of text strings

        Returns:
            List of Document objects with text chunks
        """
        if isinstance(texts, str):
            texts = [texts]

        # Create Document objects
        documents = [Document(page_content=text) for text in texts]

        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        return split_docs

    def get_document_embeddings(self, texts: Union[str, List[str]]) -> tuple[List[Document], List[List[float]]]:
        """
        Process documents and create embeddings.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Tuple of (split documents, embeddings)
        """
        # Process and split documents
        split_docs = self.process_documents(texts)

        # Get text content from split documents
        split_texts = [doc.page_content for doc in split_docs]

        # Create embeddings for split texts
        embeddings = self.create_embeddings(split_texts)

        return split_docs, embeddings

    def store_documents(self, documents: List[Document]) -> List[str]:
        """
        Store documents in Qdrant with their embeddings and metadata.
        
        Args:
            documents: List of Document objects to store
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
            
        # Generate unique IDs for documents
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        
        # Extract metadatas from documents
        metadatas = []
        for doc in documents:
            # Ensure metadata is properly structured
            metadata = doc.metadata.copy() if doc.metadata else {}
            
            # Add document ID to metadata if not present
            if "doc_id" not in metadata:
                metadata["doc_id"] = str(uuid.uuid4())
                
            # Ensure chunk_id is present
            if "chunk_id" not in metadata:
                metadata["chunk_id"] = f"chunk_{uuid.uuid4()}"
                
            # Add content type if not present
            if "content_type" not in metadata:
                metadata["content_type"] = "text"
                
            metadatas.append(metadata)
        
        # Store documents in Qdrant using LangChain's vectorstore
        self.vectorstore.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=doc_ids
        )
        
        return doc_ids

    def process_documents_to_embeddings(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process a list of Document objects and create embeddings for them.
        
        Args:
            documents: List of Document objects to process
            
        Returns:
            Dictionary with processed chunks and embeddings
        """
        if not documents:
            return {"processed_chunks": [], "error": "No documents provided"}
        
        processed_chunks = []
        
        # Process each document
        for i, doc in enumerate(tqdm(documents, desc="Processing documents")):
            try:
                # Create embedding for document
                embedding = self.embeddings.embed_query(doc.page_content)
                
                # Create processed chunk with metadata
                chunk = {
                    "id": str(uuid.uuid4()),
                    "index": i,
                    "text": doc.page_content,
                    "embedding": embedding,
                    "metadata": doc.metadata
                }
                
                # Add document to processed chunks
                processed_chunks.append(chunk)
                
            except Exception as e:
                # Log error and continue with next document
                print(f"Error processing document {i}: {str(e)}")
        
        return {
            "processed_chunks": processed_chunks,
            "chunk_count": len(processed_chunks),
            "embedding_model": "text-embedding-3-small"
        }

    def search_similar(self, query: str, k: int = 5, filter_condition: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_condition: Optional filter condition for Qdrant
            
        Returns:
            List of similar documents
        """
        if filter_condition:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_condition
            )
        else:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k
            )
            
        return results

    def search_with_score(self, query: str, k: int = 5, filter_condition: Optional[Dict] = None) -> List[tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_condition: Optional filter condition for Qdrant
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        if filter_condition:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_condition
            )
        else:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
        return results

    def update_document(self, doc_id: str, new_content: str, new_metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing document in Qdrant.
        
        Args:
            doc_id: ID of the document to update
            new_content: New content for the document
            new_metadata: Optional new metadata for the document
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Create new document with updated content
            new_doc = Document(
                page_content=new_content,
                metadata=new_metadata or {}
            )
            
            # Update document in Qdrant
            self.vectorstore.update_document(
                doc_id=doc_id,
                document=new_doc
            )
            
            return True
        except Exception as e:
            print(f"Error updating document: {str(e)}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from Qdrant.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            self.vectorstore.delete_document(doc_id)
            return True
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False

    def process_json_file(self, json_file_path: str) -> Dict[str, Any]:
        """
        Process a JSON file with parsed documents and create embeddings for its content.
        
        Args:
            json_file_path: Path to the JSON file with parsed documents
            
        Returns:
            Dictionary with processed documents and their embeddings
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            json.JSONDecodeError: If the JSON file is invalid
        """
        try:
            # Read and parse JSON file
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            result = {
                "document_count": data.get("document_count", 0),
                "total_chunk_count": data.get("total_chunk_count", 0),
                "processed_chunks": []
            }
            
            # Extract chunks from processed documents
            chunks = []
            chunk_metadata = []
            
            # Look for advanced chunking format first (chunks array at top level)
            if "chunks" in data and isinstance(data["chunks"], list):
                print(f"Found top-level chunks array with {len(data['chunks'])} items")
                for i, chunk in enumerate(data["chunks"]):
                    if isinstance(chunk, dict) and "content" in chunk and isinstance(chunk["content"], str):
                        chunk_text = chunk["content"]
                        chunks.append(chunk_text)
                        
                        # Extract metadata
                        metadata = {
                            "chunk_id": chunk.get("id", f"chunk_{i}"),
                            "doc_id": chunk.get("document_id", "unknown"),
                            "chunk_type": chunk.get("chunk_type", "unknown"),
                        }
                        
                        # Copy additional metadata if available
                        if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                            for key, value in chunk["metadata"].items():
                                if key not in ["chunk_id", "doc_id"]:  # Avoid duplicates
                                    metadata[key] = value
                            
                            # Special handling for source
                            if "source" in chunk["metadata"]:
                                metadata["source"] = str(chunk["metadata"]["source"])
                            
                            # Extract skills if available
                            if "skill_extraction" in chunk["metadata"]:
                                metadata["skills"] = chunk["metadata"]["skill_extraction"].get("validated_skills", [])
                        
                        chunk_metadata.append(metadata)
            
            # If no chunks found in "chunks" array, check processed_documents
            if not chunks and "processed_documents" in data and isinstance(data["processed_documents"], list):
                print(f"Found processed_documents with {len(data['processed_documents'])} items")
                
                # Print the first document structure to understand it
                if len(data["processed_documents"]) > 0:
                    first_doc = data["processed_documents"][0]
                    print(f"First document keys: {list(first_doc.keys() if isinstance(first_doc, dict) else [])}")
                    
                    for key in first_doc.keys():
                        try:
                            value = first_doc[key]
                            value_type = type(value).__name__
                            value_summary = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                            print(f"  {key}: {value_type} = {value_summary}")
                        except Exception as e:
                            print(f"  Error getting {key}: {str(e)}")
                
                # Iterate through processed documents to find text content
                for i, doc in enumerate(data["processed_documents"]):
                    if isinstance(doc, dict):
                        # Try to find text content in all possible fields
                        doc_text = None
                        
                        # Look for content in common fields
                        for field in ["content", "text", "raw_text", "extracted_text"]:
                            if field in doc and isinstance(doc[field], str):
                                doc_text = doc[field]
                                print(f"Found text in field '{field}' for document {i}")
                                break
                        
                        # If not found, look in data or metadata fields
                        if not doc_text and "data" in doc and isinstance(doc["data"], dict):
                            for field in ["content", "text"]:
                                if field in doc["data"] and isinstance(doc["data"][field], str):
                                    doc_text = doc["data"][field]
                                    print(f"Found text in data.{field} for document {i}")
                                    break
                        
                        # If still not found, check raw_content or document_data
                        if not doc_text and "raw_content" in doc and isinstance(doc["raw_content"], str):
                            doc_text = doc["raw_content"]
                            print(f"Found text in raw_content for document {i}")
                        
                        if not doc_text and "document_data" in doc and isinstance(doc["document_data"], dict):
                            for field in ["content", "text", "raw_text"]:
                                if field in doc["document_data"] and isinstance(doc["document_data"][field], str):
                                    doc_text = doc["document_data"][field]
                                    print(f"Found text in document_data.{field} for document {i}")
                                    break
                        
                        # Add the document text if found
                        if doc_text:
                            chunks.append(doc_text)
                            metadata = {
                                "doc_id": doc.get("id", f"doc_{i}"),
                                "source": str(doc.get("metadata", {}).get("source", f"source_{i}")),
                            }
                            # Copy additional metadata if available
                            if "metadata" in doc and isinstance(doc["metadata"], dict):
                                if "skill_extraction" in doc["metadata"]:
                                    metadata["skills"] = doc["metadata"]["skill_extraction"].get("validated_skills", [])
                                if "source" in doc["metadata"]:
                                    metadata["source"] = str(doc["metadata"]["source"])
                            chunk_metadata.append(metadata)
            
            # If we still don't have chunks, try to extract text from the raw document content
            if not chunks and "processed_documents" in data and isinstance(data["processed_documents"], list):
                print("No direct text content found in processed_documents, trying to extract from nested properties")
                
                # Use the text splitter to create chunks from any relevant text we can find
                raw_texts = []
                for i, doc in enumerate(data["processed_documents"]):
                    # Collect all text fields
                    all_text = ""
                    if isinstance(doc, dict):
                        # Try extracting from all properties that might contain text
                        for key, value in doc.items():
                            if key != "metadata" and isinstance(value, str) and len(value) > 50:
                                all_text += value + "\n\n"
                    
                    if all_text:
                        raw_texts.append(all_text)
                
                if raw_texts:
                    print(f"Extracted {len(raw_texts)} raw text documents")
                    # Use text splitter to create chunks from the raw texts
                    for i, raw_text in enumerate(raw_texts):
                        doc_chunks = self.text_splitter.split_text(raw_text)
                        for j, chunk_text in enumerate(doc_chunks):
                            chunks.append(chunk_text)
                            metadata = {
                                "doc_id": f"doc_{i}",
                                "chunk_id": f"manual_chunk_{j}",
                                "source": "manual_splitting"
                            }
                            chunk_metadata.append(metadata)
                    
                    print(f"Created {len(chunks)} manual chunks from {len(raw_texts)} documents")
            
            if not chunks:
                # Print the keys to help debug
                print(f"Available top-level keys: {list(data.keys())}")
                raise ValueError("No chunks/text content found in the JSON file. Please check file structure.")
            
            print(f"Found {len(chunks)} text chunks to process")
            
            # Create embeddings for all chunks
            embeddings = self.create_embeddings(chunks)
            
            # Combine chunks, metadata and embeddings
            for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, chunk_metadata)):
                result["processed_chunks"].append({
                    "id": i,
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": metadata
                })
            
            return result
            
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON file: {json_file_path}", "", 0)
        except Exception as e:
            raise Exception(f"Error processing JSON file: {str(e)}")

    def save_embeddings(self, embeddings_result: Dict[str, Any], output_path: str) -> str:
        """
        Save embeddings to a JSON file.
        
        Args:
            embeddings_result: Dictionary with processed chunks and embeddings
            output_path: Path to save the embeddings
            
        Returns:
            Path to the saved embeddings file
        """
        # Convert numpy arrays to lists for JSON serialization
        processed_chunks = embeddings_result.get("processed_chunks", [])
        for chunk in processed_chunks:
            if "embedding" in chunk and hasattr(chunk["embedding"], "tolist"):
                chunk["embedding"] = chunk["embedding"].tolist()
        
        # Save to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_result, f, indent=2, ensure_ascii=False)
            
        return output_path

    def load_and_store_embeddings(self, json_file_path: str, collection_name: str = "document_embeddings", batch_size: int = 100) -> Dict[str, Any]:
        """
        Load embeddings from a JSON file and store them in Qdrant.
        
        Args:
            json_file_path: Path to the JSON file with embeddings
            collection_name: Name of the Qdrant collection to store embeddings in
            batch_size: Number of embeddings to insert in a single batch
            
        Returns:
            Dictionary with results (embeddings inserted, elapsed time, etc.)
        """
        import time
        
        # Start timing
        start_time = time.time()
        
        # Load embeddings from JSON
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract processed chunks
        processed_chunks = data.get("processed_chunks", [])
        
        # Check if embeddings exist
        if not processed_chunks:
            return {
                "embeddings_inserted": 0,
                "elapsed_seconds": 0,
                "error": "No embeddings found in JSON file"
            }
        
        # Ensure Qdrant collection exists
        try:
            # Try to get collection info
            collection_info = self.qdrant_client.get_collection(collection_name)
            
            # Check vector size from first embedding
            vector_size = len(processed_chunks[0]["embedding"])
            
            # Check if vector size matches
            if collection_info.config.params.vectors.size != vector_size:
                # Delete and recreate with correct size
                self.qdrant_client.delete_collection(collection_name)
                raise ValueError("Collection exists with wrong vector size")
                
        except Exception:
            # Get vector size from first embedding
            vector_size = len(processed_chunks[0]["embedding"])
            
            # Create collection with proper configuration
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created '{collection_name}' collection in Qdrant with vector size {vector_size}")
        
        # Prepare batches for insertion
        total_embeddings = len(processed_chunks)
        batches = [processed_chunks[i:i + batch_size] for i in range(0, total_embeddings, batch_size)]
        
        # Track insertion progress
        embeddings_inserted = 0
        
        # Process each batch
        for batch_index, batch in enumerate(tqdm(batches, desc=f"Storing embeddings in {collection_name}")):
            try:
                # Prepare data for batch insertion
                ids = []
                vectors = []
                payloads = []
                
                for chunk in batch:
                    # Generate unique ID if not present
                    chunk_id = chunk.get("id", str(uuid.uuid4()))
                    ids.append(chunk_id)
                    
                    # Add vector
                    vectors.append(chunk["embedding"])
                    
                    # Prepare payload (metadata + text)
                    # IMPORTANT: Store text in the page_content field for LangChain compatibility
                    payload = {
                        "page_content": chunk["text"],  # Use page_content instead of text
                        "metadata": chunk.get("metadata", {})
                    }
                    payloads.append(payload)
                
                # Insert batch
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=models.Batch(
                        ids=ids,
                        vectors=vectors,
                        payloads=payloads
                    )
                )
                
                # Update count
                embeddings_inserted += len(batch)
                
            except Exception as e:
                print(f"Error inserting batch {batch_index}: {str(e)}")
        
        # Calculate elapsed time
        elapsed_seconds = time.time() - start_time
        
        # Return results
        return {
            "collection_name": collection_name,
            "total_embeddings": total_embeddings,
            "embeddings_inserted": embeddings_inserted,
            "batch_size_used": batch_size,
            "elapsed_seconds": elapsed_seconds
        }

    def load_embeddings_from_json(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load embeddings from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file with embeddings
            
        Returns:
            Dictionary with loaded embeddings data
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Embeddings file not found: {json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in embeddings file: {json_file_path}")
    
    def create_qdrant_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Create a Qdrant collection with specified vector size and cosine similarity.
        
        Args:
            collection_name: Name of the collection to create
            vector_size: Dimension of the embedding vectors
        """
        # Check if collection already exists
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name in collection_names:
            print(f"Collection '{collection_name}' already exists.")
            return
        
        # Create the collection with the specified vector size
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
        )
        print(f"Created collection '{collection_name}' with vector size {vector_size}")
    
    def store_embeddings_in_qdrant(
        self, 
        embeddings_data: Dict[str, Any], 
        collection_name: str = "document_embeddings",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Store embeddings from JSON data into a Qdrant collection with optimized batch insertion.
        
        Args:
            embeddings_data: Dictionary with embeddings data loaded from JSON
            collection_name: Name of the Qdrant collection to store embeddings in
            batch_size: Number of embeddings to insert in a single batch
            
        Returns:
            Dictionary with insertion statistics
        """
        start_time = time.time()
        processed_chunks = embeddings_data.get("processed_chunks", [])
        
        if not processed_chunks:
            raise ValueError("No embeddings found in the provided data")
        
        # Get the vector size from the first embedding
        if processed_chunks and "embedding" in processed_chunks[0]:
            vector_size = len(processed_chunks[0]["embedding"])
            print(f"Detected vector size: {vector_size}")
            
            # Create the collection if it doesn't exist
            self.create_qdrant_collection(collection_name, vector_size)
        else:
            raise ValueError("No valid embeddings found in the data")
        
        # Prepare batches for insertion
        total_chunks = len(processed_chunks)
        batches = [processed_chunks[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
        
        print(f"Inserting {total_chunks} embeddings in {len(batches)} batches...")
        
        # Process each batch
        total_inserted = 0
        for i, batch in enumerate(tqdm(batches, desc="Storing embeddings")):
            points = []
            
            for chunk in batch:
                # Extract data and create a point
                try:
                    point_id = chunk["id"]
                    embedding = chunk["embedding"]
                    
                    # Prepare payload with structured metadata
                    # IMPORTANT: Store text in the page_content field for LangChain compatibility
                    payload = {
                        "page_content": chunk["text"],  # Use page_content instead of text
                        "metadata": chunk.get("metadata", {})
                    }
                    
                    # Add point to batch
                    points.append(models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    ))
                except KeyError as e:
                    print(f"Warning: Missing required field in chunk {i}: {e}")
                    continue
            
            # Insert batch if not empty
            if points:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                total_inserted += len(points)
                
                # Print progress periodically
                if (i + 1) % 5 == 0 or i == len(batches) - 1:
                    elapsed = time.time() - start_time
                    print(f"Progress: {total_inserted}/{total_chunks} embeddings inserted ({elapsed:.2f}s elapsed)")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Return insertion statistics
        stats = {
            "collection_name": collection_name,
            "total_embeddings": total_chunks,
            "embeddings_inserted": total_inserted,
            "elapsed_seconds": elapsed_time,
            "batch_size_used": batch_size
        }
        
        print(f"\nInsertion complete!")
        print(f"Inserted {total_inserted} embeddings in {elapsed_time:.2f} seconds")
        
        return stats