"""
Embeddings Module using Hugging Face Sentence Transformers
Handles text embedding generation for the RAG pipeline.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manages text embeddings using Hugging Face Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the sentence transformer model to use
                      Default: 'all-MiniLM-L6-v2' (384 dimensions, fast and efficient)
                      Other options: 'all-mpnet-base-v2' (768 dimensions, higher quality)
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully!")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Numpy array containing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar

        Returns:
            Numpy array containing all embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension size
        """
        return self.model.get_sentence_embedding_dimension()
