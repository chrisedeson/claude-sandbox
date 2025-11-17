# RAG Pipeline with Hugging Face and ChromaDB

A complete Retrieval Augmented Generation (RAG) pipeline with a beautiful Gradio UI for document question answering.

## Features

- **Document Loading**: Automatically loads documents from a data folder
- **Sentence Transformers**: Uses Hugging Face models for high-quality embeddings
- **Vector Database**: ChromaDB for efficient similarity search with persistent storage
- **Interactive UI**: Beautiful Gradio interface for querying documents
- **Flexible Architecture**: Easy to customize and extend

## Architecture

```
┌─────────────────┐
│  Data Folder    │
│  (.txt, .md)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Loader │
│  & Text Chunker │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Sentence      │
│  Transformers   │
│   (Embeddings)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │
│ Vector Database │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Gradio UI     │
│  Query Interface│
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd claude-sandbox
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Add your documents to the `data/` folder (supports `.txt` and `.md` files)

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:7860`

4. Click "Initialize Pipeline" in the "Manage Pipeline" tab

5. Start querying your documents in the "Query Documents" tab!

### Adding New Documents

1. Place your text files in the `data/` folder
2. Go to the "Manage Pipeline" tab in the UI
3. Click "Reindex Documents"
4. Your new documents are now searchable!

## Project Structure

```
claude-sandbox/
├── data/                          # Document storage folder
│   ├── sample_doc1.txt           # Sample documents included
│   ├── sample_doc2.txt
│   └── sample_doc3.txt
├── src/                           # Source code
│   ├── __init__.py
│   ├── document_loader.py        # Document loading and chunking
│   ├── embeddings.py             # Sentence transformer embeddings
│   ├── vector_store.py           # ChromaDB vector database
│   └── rag_pipeline.py           # Main RAG pipeline orchestration
├── app.py                         # Gradio UI application
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Components

### Document Loader (`src/document_loader.py`)
- Loads documents from the data folder
- Supports `.txt` and `.md` files
- Splits text into overlapping chunks for better retrieval

### Embeddings (`src/embeddings.py`)
- Uses Hugging Face Sentence Transformers
- Default model: `all-MiniLM-L6-v2` (384 dimensions, fast and efficient)
- Batch processing for efficient embedding generation

### Vector Store (`src/vector_store.py`)
- ChromaDB for vector storage and similarity search
- Persistent storage (survives application restarts)
- Cosine similarity for document matching

### RAG Pipeline (`src/rag_pipeline.py`)
- Orchestrates all components
- Handles document indexing and querying
- Generates context for LLM integration

### Gradio UI (`app.py`)
- Interactive web interface
- Document querying with adjustable result count
- Pipeline management (initialization, reindexing, statistics)
- Example queries for quick testing

## Configuration

You can customize the pipeline by modifying the initialization parameters in `app.py`:

```python
pipeline = RAGPipeline(
    data_folder="data",              # Path to documents
    model_name="all-MiniLM-L6-v2",  # Embedding model
    collection_name="rag_documents",  # ChromaDB collection
    chunk_size=500,                   # Characters per chunk
    chunk_overlap=50                  # Overlap between chunks
)
```

### Available Embedding Models

- `all-MiniLM-L6-v2`: Fast, 384 dimensions (default)
- `all-mpnet-base-v2`: Higher quality, 768 dimensions
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for question answering

## Use Cases

1. **Document Question Answering**: Ask questions about your document collection
2. **Knowledge Base Search**: Find relevant information quickly
3. **Context Generation**: Generate context for LLM prompts
4. **Information Retrieval**: Semantic search across documents

## Advanced Usage

### Programmatic Usage

You can use the RAG pipeline programmatically:

```python
from src.rag_pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline(data_folder="data")

# Index documents
pipeline.index_documents()

# Query
results = pipeline.query("What is machine learning?", top_k=3)

# Generate context for LLM
context = pipeline.generate_context("What is machine learning?", top_k=3)
```

### Integration with LLMs

The pipeline generates formatted context that can be used with any LLM:

```python
context = pipeline.generate_context("Your question here", top_k=3)
prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: Your question here

Answer:"""
```

## Performance

- **Embedding Model**: all-MiniLM-L6-v2 processes ~1000 sentences/second on CPU
- **Vector Search**: ChromaDB provides fast similarity search even with large collections
- **Memory**: ~500MB base memory + ~1MB per 1000 document chunks

## Troubleshooting

### Issue: "No documents found"
- Ensure your files are in the `data/` folder
- Check that files have `.txt` or `.md` extensions
- Verify file permissions

### Issue: "Out of memory"
- Reduce `chunk_size` in the pipeline configuration
- Process documents in smaller batches
- Use a smaller embedding model

### Issue: "Slow embedding generation"
- The first run downloads the model (~90MB)
- Consider using GPU acceleration by installing `torch` with CUDA support

## Contributing

Feel free to open issues or submit pull requests for improvements!

## License

MIT License

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector database
- [Gradio](https://gradio.app/) for the UI framework
