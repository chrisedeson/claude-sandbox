"""
Quick demo script to test the RAG pipeline.
Run this after installing dependencies: pip install -r requirements.txt
"""

def main():
    print("=" * 70)
    print("RAG Pipeline Demo - Testing End-to-End")
    print("=" * 70)

    try:
        # Import the pipeline
        print("\n[1/5] Importing RAG pipeline...")
        from src.rag_pipeline import RAGPipeline
        print("      ✓ Import successful")

        # Initialize
        print("\n[2/5] Initializing pipeline...")
        print("      (This may take a moment on first run - downloading model)")
        pipeline = RAGPipeline(
            data_folder="data",
            model_name="all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=50
        )
        print("      ✓ Pipeline initialized")

        # Index documents
        print("\n[3/5] Indexing documents from data/ folder...")
        pipeline.index_documents()
        print("      ✓ Documents indexed")

        # Show statistics
        print("\n[4/5] Pipeline Statistics:")
        stats = pipeline.get_stats()
        print(f"      • Document chunks: {stats['document_count']}")
        print(f"      • Embedding model: {stats['model_name']}")
        print(f"      • Embedding dimension: {stats['embedding_dimension']}")
        print(f"      • Status: {'Indexed ✓' if stats['is_indexed'] else 'Not indexed'}")

        # Test queries
        print("\n[5/5] Testing queries...")
        test_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What is NLP used for?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n      Query {i}: '{query}'")
            results = pipeline.query(query, top_k=2)

            if results:
                for j, result in enumerate(results, 1):
                    similarity = 1 - result['distance']
                    filename = result['metadata']['filename']
                    preview = result['content'][:100].replace('\n', ' ')

                    print(f"        Result {j}: {filename}")
                    print(f"        Similarity: {similarity:.3f}")
                    print(f"        Preview: {preview}...")
            else:
                print("        No results found")

        # Generate context example
        print("\n" + "=" * 70)
        print("Context Generation Example")
        print("=" * 70)
        query = "What is machine learning?"
        context = pipeline.generate_context(query, top_k=2)
        print(f"\nQuery: {query}")
        print("\nGenerated Context (for use with LLMs):")
        print("-" * 70)
        print(context[:500] + "..." if len(context) > 500 else context)
        print("-" * 70)

        print("\n" + "=" * 70)
        print("✓ Demo completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  • Run 'python app.py' to start the Gradio UI")
        print("  • Add your own documents to the data/ folder")
        print("  • Access the UI at http://localhost:7860")
        print("\nTo customize:")
        print("  • Edit data/ folder with your documents")
        print("  • Modify chunk_size in RAGPipeline initialization")
        print("  • Try different embedding models (see README.md)")

    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt")
        return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
