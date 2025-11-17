# Testing Guide for RAG Pipeline

This guide provides multiple ways to test the RAG pipeline, from quick validation to full end-to-end testing.

## Test 1: Quick Validation (No Dependencies) ✓

This test verifies the structure without installing any packages:

```bash
python test_quick.py
```

**What it checks:**
- All files are present
- Python syntax is valid
- Sample documents can be read

## Test 2: Full Application Test (Recommended)

This is the complete test with the Gradio UI.

### Step 1: Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**Note:** First installation will download:
- Sentence Transformer model (~90MB)
- PyTorch and other libraries (~2GB total)

### Step 2: Run the Application

```bash
python app.py
```

You should see:
```
Starting RAG Pipeline Application...
Loading embedding model: all-MiniLM-L6-v2
Embedding model loaded successfully!
Vector store initialized: rag_documents
...
Running on local URL:  http://0.0.0.0:7860
```

### Step 3: Test in Browser

1. Open `http://localhost:7860` in your browser

2. **Initialize the Pipeline:**
   - Go to "Manage Pipeline" tab
   - Click "Initialize Pipeline"
   - Wait for "✓ Pipeline initialized successfully!"
   - Click "Show Statistics" to verify documents are indexed

3. **Query Documents:**
   - Go to "Query Documents" tab
   - Try these example queries:
     - "What is machine learning?"
     - "Explain neural networks"
     - "What are NLP applications?"
   - Verify that relevant documents are returned
   - Check the similarity scores (should be > 0.5 for good matches)

4. **Test with Custom Query:**
   - Ask your own question related to the sample documents
   - Adjust "Number of documents to retrieve" slider
   - Verify results update accordingly

### Expected Results

For query "What is machine learning?":
- Should return chunks from `sample_doc1.txt` with high similarity
- Similarity score should be > 0.7
- Context should be generated in the right panel

## Test 3: Programmatic Test

Create a test script to verify the pipeline programmatically:

```bash
# Create and run this test
cat > test_pipeline.py << 'EOF'
"""Test RAG pipeline programmatically."""
from src.rag_pipeline import RAGPipeline

print("Initializing RAG Pipeline...")
pipeline = RAGPipeline(data_folder="data")

print("Indexing documents...")
pipeline.index_documents()

print("\nPipeline Statistics:")
stats = pipeline.get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

print("\nTesting query: 'What is machine learning?'")
results = pipeline.query("What is machine learning?", top_k=3)

print(f"\nFound {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  File: {result['metadata']['filename']}")
    print(f"  Similarity: {1 - result['distance']:.3f}")
    print(f"  Preview: {result['content'][:100]}...")

print("\n✓ Pipeline test completed successfully!")
EOF

python test_pipeline.py
```

## Test 4: Add Your Own Documents

Test with your own data:

```bash
# Add a new document
cat > data/custom_test.txt << 'EOF'
This is a custom test document about quantum computing.
Quantum computers use quantum bits or qubits instead of classical bits.
They can solve certain problems much faster than classical computers.
EOF

# Reindex and test
python -c "
from src.rag_pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.index_documents(force_reindex=True)
results = pipeline.query('What is quantum computing?', top_k=1)
print('Query:', 'What is quantum computing?')
print('Result:', results[0]['content'][:200])
print('Source:', results[0]['metadata']['filename'])
"
```

## Test 5: Performance Test

Test with many documents:

```bash
# Generate test documents
python << 'EOF'
import os
os.makedirs('data/test_docs', exist_ok=True)

topics = ['AI', 'Physics', 'Biology', 'Chemistry', 'Math']
for i in range(20):
    topic = topics[i % len(topics)]
    with open(f'data/test_docs/doc_{i:02d}.txt', 'w') as f:
        f.write(f"Document about {topic}. " * 50)

print("Created 20 test documents")
EOF

# Run performance test
python << 'EOF'
import time
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

print("Indexing documents...")
start = time.time()
pipeline.index_documents(force_reindex=True)
index_time = time.time() - start

stats = pipeline.get_stats()
print(f"Indexed {stats['document_count']} chunks in {index_time:.2f}s")

print("\nTesting query performance...")
start = time.time()
results = pipeline.query("Tell me about AI", top_k=5)
query_time = time.time() - start

print(f"Query returned {len(results)} results in {query_time:.3f}s")
EOF
```

## Troubleshooting Tests

### Issue: Import errors during testing
**Solution:** Install dependencies first:
```bash
pip install -r requirements.txt
```

### Issue: "No module named 'sentence_transformers'"
**Solution:** The model needs to be installed:
```bash
pip install sentence-transformers
```

### Issue: First run is very slow
**Expected:** First run downloads the embedding model (~90MB). Subsequent runs are much faster.

### Issue: Out of memory
**Solution:** Reduce chunk size or use smaller model:
```python
pipeline = RAGPipeline(
    chunk_size=200,  # Smaller chunks
    model_name="all-MiniLM-L6-v2"  # Already the smallest
)
```

## Success Criteria

Your RAG pipeline is working correctly if:

1. ✓ Quick test passes (structure validation)
2. ✓ Application starts without errors
3. ✓ Documents are indexed successfully
4. ✓ Queries return relevant results with high similarity scores (> 0.5)
5. ✓ New documents can be added and reindexed
6. ✓ UI is accessible and responsive

## Next Steps After Testing

Once all tests pass:

1. Add your own documents to the `data/` folder
2. Reindex documents in the UI
3. Start querying your data
4. Integrate with LLMs using the generated context
5. Customize the pipeline parameters for your use case

## Automated Test Script

Run all quick tests at once:

```bash
# Run comprehensive test suite
python test_quick.py && echo -e "\n✓ All tests passed! Ready to install dependencies."
```
