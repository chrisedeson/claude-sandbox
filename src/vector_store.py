"""
Vector Store Module using ChromaDB
Handles storing and retrieving document embeddings.
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import numpy as np


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""

    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        print(f"Vector store initialized: {collection_name}")
        print(f"Current document count: {self.collection.count()}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents with their embeddings to the vector store.

        Args:
            documents: List of document texts
            embeddings: Numpy array of embeddings
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of unique IDs for each document
        """
        # Generate IDs if not provided
        if ids is None:
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]

        # Convert numpy array to list for ChromaDB
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas if metadatas else [{} for _ in documents],
            ids=ids
        )

        print(f"Added {len(documents)} documents to vector store")

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5
    ) -> Dict:
        """
        Search for similar documents using query embedding.

        Args:
            query_embedding: Embedding vector of the query
            n_results: Number of results to return

        Returns:
            Dictionary containing documents, distances, and metadata
        """
        # Convert numpy array to list for ChromaDB
        query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results
        )

        return results

    def clear_collection(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Cleared collection: {self.collection_name}")

    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.

        Returns:
            Number of documents
        """
        return self.collection.count()

    def delete_by_ids(self, ids: List[str]):
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        print(f"Deleted {len(ids)} documents")
