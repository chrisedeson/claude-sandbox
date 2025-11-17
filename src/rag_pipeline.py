"""
RAG Pipeline Module
Main pipeline that orchestrates document loading, embedding, and retrieval.
"""
from typing import List, Dict, Optional
from .document_loader import DocumentLoader, TextChunker
from .embeddings import EmbeddingManager
from .vector_store import VectorStore


class RAGPipeline:
    """Main RAG Pipeline that orchestrates all components."""

    def __init__(
        self,
        data_folder: str = "data",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "rag_documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the RAG pipeline.

        Args:
            data_folder: Path to folder containing documents
            model_name: Sentence transformer model name
            collection_name: ChromaDB collection name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        print("Initializing RAG Pipeline...")

        self.document_loader = DocumentLoader(data_folder)
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding_manager = EmbeddingManager(model_name)
        self.vector_store = VectorStore(collection_name)

        self.is_indexed = False

        print("RAG Pipeline initialized successfully!")

    def index_documents(self, force_reindex: bool = False):
        """
        Load, chunk, embed, and store documents from the data folder.

        Args:
            force_reindex: If True, clear existing index and reindex all documents
        """
        if self.is_indexed and not force_reindex:
            print("Documents already indexed. Use force_reindex=True to reindex.")
            return

        if force_reindex:
            print("Clearing existing index...")
            self.vector_store.clear_collection()

        print("Loading documents...")
        documents = self.document_loader.load_documents()

        if not documents:
            print("No documents found in the data folder!")
            return

        print(f"Loaded {len(documents)} documents")

        print("Chunking documents...")
        chunked_docs = self.text_chunker.chunk_documents(documents)
        print(f"Created {len(chunked_docs)} chunks")

        print("Generating embeddings...")
        texts = [doc['content'] for doc in chunked_docs]
        embeddings = self.embedding_manager.embed_texts(texts)

        print("Storing in vector database...")
        metadatas = [
            {
                'source': doc['source'],
                'filename': doc['filename'],
                'chunk_id': doc['chunk_id']
            }
            for doc in chunked_docs
        ]

        self.vector_store.add_documents(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        self.is_indexed = True
        print("Indexing complete!")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Query the RAG pipeline to retrieve relevant documents.

        Args:
            query_text: The search query
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        if not self.is_indexed and self.vector_store.get_document_count() == 0:
            raise ValueError("No documents indexed. Please run index_documents() first.")

        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query_text)

        # Search vector store
        results = self.vector_store.search(query_embedding, n_results=top_k)

        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

        return formatted_results

    def generate_context(self, query_text: str, top_k: int = 3) -> str:
        """
        Generate context string from retrieved documents for RAG.

        Args:
            query_text: The search query
            top_k: Number of top results to retrieve

        Returns:
            Formatted context string
        """
        results = self.query(query_text, top_k)

        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i} - {result['metadata'].get('filename', 'Unknown')}]")
            context_parts.append(result['content'])
            context_parts.append("")  # Empty line for separation

        return "\n".join(context_parts)

    def get_stats(self) -> Dict:
        """
        Get statistics about the indexed documents.

        Returns:
            Dictionary containing pipeline statistics
        """
        return {
            'document_count': self.vector_store.get_document_count(),
            'embedding_dimension': self.embedding_manager.get_embedding_dimension(),
            'model_name': self.embedding_manager.model_name,
            'is_indexed': self.is_indexed
        }
