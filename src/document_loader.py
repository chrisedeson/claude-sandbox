"""
Document Loader and Text Chunking Module
Handles loading documents from the data folder and splitting them into chunks.
"""
import os
from typing import List, Dict
from pathlib import Path


class DocumentLoader:
    """Loads and processes documents from a specified directory."""

    def __init__(self, data_folder: str = "data"):
        """
        Initialize the document loader.

        Args:
            data_folder: Path to the folder containing documents
        """
        self.data_folder = Path(data_folder)

    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load all text documents from the data folder.

        Returns:
            List of dictionaries containing document content and metadata
        """
        documents = []

        if not self.data_folder.exists():
            raise ValueError(f"Data folder '{self.data_folder}' does not exist")

        # Supported file extensions
        supported_extensions = ['.txt', '.md']

        for file_path in self.data_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix in supported_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    documents.append({
                        'content': content,
                        'source': str(file_path),
                        'filename': file_path.name
                    })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return documents


class TextChunker:
    """Splits text into chunks for embedding."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence ending punctuation
                last_period = max(
                    chunk.rfind('. '),
                    chunk.rfind('! '),
                    chunk.rfind('? ')
                )
                if last_period > self.chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

            if start >= len(text):
                break

        return [c for c in chunks if c]  # Remove empty chunks

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chunk multiple documents while preserving metadata.

        Args:
            documents: List of document dictionaries

        Returns:
            List of chunk dictionaries with metadata
        """
        chunked_docs = []

        for doc in documents:
            chunks = self.chunk_text(doc['content'])

            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    'content': chunk,
                    'source': doc['source'],
                    'filename': doc['filename'],
                    'chunk_id': i
                })

        return chunked_docs
