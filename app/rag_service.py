#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides RAG (Retrieval-Augmented Generation) functionality
using Qdrant for vector storage and retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

class RAGService:
    def __init__(
        self,
        collection_name: str = "document_embeddings",
        host: str = "localhost",
        port: int = 6333,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt4o",
        max_tokens: int = 120000  # Conservative limit to stay within bounds
    ):
        """
        Initialize the RAG service.
        
        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant server host
            port: Qdrant server port
            embedding_model: Name of the OpenAI embedding model
            llm_model: Name of the OpenAI LLM model
            max_tokens: Maximum number of tokens to process
        """
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=host, port=port)
        
        # Initialize vectorstore
        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )
        
        # Set max tokens
        self.max_tokens = max_tokens
        
        # Create the RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _create_rag_chain(self):
        """Create the RAG chain using the newer LangChain components."""
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer: """),
        ])
        
        # Create the chain
        chain = (
            {"context": self.vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    
    def _truncate_context(self, context: str, max_context_tokens: int = 100000) -> str:
        """Truncate context to stay within token limits."""
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(context)
        
        if len(tokens) <= max_context_tokens:
            return context
            
        # Truncate to max tokens while preserving complete sentences
        truncated_tokens = tokens[:max_context_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        
        # Find the last complete sentence
        last_period = truncated_text.rfind('.')
        if last_period > 0:
            truncated_text = truncated_text[:last_period + 1]
            
        return truncated_text
    
    def query(self, question: str, k: int = 5, filter_condition: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to answer
            k: Number of documents to retrieve
            filter_condition: Optional filter condition for Qdrant
            
        Returns:
            Dictionary containing answer and source documents
        """
        try:
            # Check question length
            question_tokens = self._count_tokens(question)
            if question_tokens > self.max_tokens:
                return {
                    "error": "Question is too long. Please make it shorter.",
                    "answer": None,
                    "source_documents": []
                }
            
            # Update retriever parameters if needed
            if k != 5 or filter_condition:
                self.rag_chain = self._create_rag_chain()
            
            # Get source documents first
            source_docs = self.vectorstore.similarity_search(
                question,
                k=k,
                filter=filter_condition
            )
            
            # Combine context from source documents
            context = "\n\n".join(doc.page_content for doc in source_docs)
            
            # Truncate context if needed
            context = self._truncate_context(context)
            
            # Get answer
            answer = self.rag_chain.invoke(question)
            
            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("score", 0.0)
                    }
                    for doc in source_docs
                ]
            }
            
        except Exception as e:
            return {
                "error": f"Error processing query: {str(e)}",
                "answer": None,
                "source_documents": []
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
        try:
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
        except Exception as e:
            print(f"Error in search_similar: {str(e)}")
            return []
    
    def search_with_score(self, query: str, k: int = 5, filter_condition: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_condition: Optional filter condition for Qdrant
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        try:
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
        except Exception as e:
            print(f"Error in search_with_score: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        try:
            # Search for document with exact ID match
            results = self.vectorstore.similarity_search(
                query="",  # Empty query since we're filtering by ID
                k=1,
                filter={"doc_id": doc_id}
            )
            
            if results:
                return results[0]
            return None
            
        except Exception as e:
            print(f"Error retrieving document: {str(e)}")
            return None 