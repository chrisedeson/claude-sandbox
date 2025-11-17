"""
Quick test to verify RAG pipeline without heavy dependencies.
Tests basic structure and document loading.
"""
import sys
import os

def test_file_structure():
    """Verify all files exist."""
    print("Testing file structure...")

    files = [
        "app.py", "requirements.txt", "README.md",
        "src/__init__.py", "src/document_loader.py",
        "src/embeddings.py", "src/vector_store.py",
        "src/rag_pipeline.py",
        "data/sample_doc1.txt", "data/sample_doc2.txt", "data/sample_doc3.txt"
    ]

    for f in files:
        exists = "✓" if os.path.exists(f) else "✗"
        print(f"  {exists} {f}")

    return all(os.path.exists(f) for f in files)

def test_document_loading_simple():
    """Test document loading without dependencies."""
    print("\nTesting document loading...")

    # Read sample documents directly
    for i in range(1, 4):
        path = f"data/sample_doc{i}.txt"
        try:
            with open(path, 'r') as f:
                content = f.read()
            print(f"  ✓ Loaded {path} ({len(content)} chars)")
        except Exception as e:
            print(f"  ✗ Failed to load {path}: {e}")
            return False

    return True

def test_python_syntax():
    """Test Python syntax of all source files."""
    print("\nTesting Python syntax...")

    files = [
        "app.py",
        "src/__init__.py",
        "src/document_loader.py",
        "src/embeddings.py",
        "src/vector_store.py",
        "src/rag_pipeline.py"
    ]

    for f in files:
        try:
            with open(f, 'r') as file:
                compile(file.read(), f, 'exec')
            print(f"  ✓ {f} - syntax OK")
        except SyntaxError as e:
            print(f"  ✗ {f} - syntax error: {e}")
            return False

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline - Quick Test (No Dependencies Required)")
    print("=" * 60)

    all_passed = True

    if not test_file_structure():
        print("\n✗ File structure test failed!")
        all_passed = False

    if not test_python_syntax():
        print("\n✗ Python syntax test failed!")
        all_passed = False

    if not test_document_loading_simple():
        print("\n✗ Document loading test failed!")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All quick tests passed!")
        print("\nNext steps to run the full application:")
        print("  1. pip install -r requirements.txt")
        print("  2. python app.py")
        print("  3. Open http://localhost:7860")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
    print("=" * 60)
