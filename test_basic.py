"""
Basic test script to verify the RAG pipeline structure.
This tests imports and basic structure without requiring all dependencies.
"""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.document_loader import DocumentLoader, TextChunker
        print("✓ Document loader imported successfully")
    except ImportError as e:
        print(f"✗ Document loader import failed: {e}")
        return False

    print("\nAll basic structure tests passed!")
    return True


def test_document_loading():
    """Test document loading functionality."""
    print("\nTesting document loading...")

    from src.document_loader import DocumentLoader, TextChunker

    # Test document loader
    loader = DocumentLoader("data")
    print(f"✓ DocumentLoader initialized with folder: data")

    # Test text chunker
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    print(f"✓ TextChunker initialized")

    # Test chunking
    sample_text = "This is a test. " * 100
    chunks = chunker.chunk_text(sample_text)
    print(f"✓ Text chunking works: {len(chunks)} chunks created from sample text")

    return True


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    import os
    from pathlib import Path

    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "src/__init__.py",
        "src/document_loader.py",
        "src/embeddings.py",
        "src/vector_store.py",
        "src/rag_pipeline.py",
        "data/sample_doc1.txt",
        "data/sample_doc2.txt",
        "data/sample_doc3.txt",
    ]

    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} missing!")
            all_exist = False

    return all_exist


if __name__ == "__main__":
    print("=" * 50)
    print("RAG Pipeline - Basic Structure Tests")
    print("=" * 50)

    # Test file structure
    if not test_file_structure():
        print("\n✗ File structure test failed!")
        exit(1)

    # Test imports
    if not test_imports():
        print("\n✗ Import test failed!")
        exit(1)

    # Test document loading
    if not test_document_loading():
        print("\n✗ Document loading test failed!")
        exit(1)

    print("\n" + "=" * 50)
    print("✓ All basic tests passed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the application: python app.py")
    print("3. Open http://localhost:7860 in your browser")
