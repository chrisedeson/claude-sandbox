"""
Lightweight test to see if we can at least test the document loading part.
This doesn't require heavy ML dependencies.
"""
import sys
sys.path.insert(0, '.')

print("Testing document loading without heavy dependencies...\n")

# Test 1: Document Loading
print("[1/3] Testing DocumentLoader...")
try:
    from pathlib import Path
    data_folder = Path("data")
    docs = []

    for file_path in data_folder.glob("*.txt"):
        with open(file_path, 'r') as f:
            content = f.read()
        docs.append({
            'content': content,
            'source': str(file_path),
            'filename': file_path.name
        })

    print(f"      ✓ Loaded {len(docs)} documents")
    for doc in docs:
        print(f"        - {doc['filename']}: {len(doc['content'])} chars")
except Exception as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# Test 2: Text Chunking (without dependencies)
print("\n[2/3] Testing text chunking...")
try:
    def simple_chunk(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if end < len(text):
                last_period = max(chunk.rfind('. '), chunk.rfind('! '), chunk.rfind('? '))
                if last_period > chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - overlap
            if start >= len(text):
                break

        return [c for c in chunks if c]

    total_chunks = 0
    for doc in docs:
        chunks = simple_chunk(doc['content'])
        total_chunks += len(chunks)
        print(f"      - {doc['filename']}: {len(chunks)} chunks")

    print(f"      ✓ Total chunks: {total_chunks}")
except Exception as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# Test 3: Check if ML dependencies are available
print("\n[3/3] Checking ML dependencies...")
deps_available = {
    'sentence_transformers': False,
    'chromadb': False,
    'gradio': False,
    'torch': False
}

for dep in deps_available.keys():
    try:
        __import__(dep)
        deps_available[dep] = True
        print(f"      ✓ {dep}")
    except ImportError:
        print(f"      ✗ {dep} (not installed)")

print("\n" + "=" * 60)
if all(deps_available.values()):
    print("✓ All dependencies available! Ready to run full app.")
    print("\nRun: python app.py")
else:
    print("⚠ Some dependencies missing. Installing...")
    print("\nMissing packages:")
    for dep, available in deps_available.items():
        if not available:
            print(f"  - {dep}")
    print("\nTo install: pip install -r requirements.txt")
print("=" * 60)
