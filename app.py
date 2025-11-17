"""
Gradio UI for RAG Pipeline
Interactive interface for querying documents using RAG.
"""
import gradio as gr
from src.rag_pipeline import RAGPipeline
import os


# Initialize the RAG pipeline
pipeline = None


def initialize_pipeline():
    """Initialize the RAG pipeline and index documents."""
    global pipeline
    try:
        pipeline = RAGPipeline(
            data_folder="data",
            model_name="all-MiniLM-L6-v2",
            collection_name="rag_documents",
            chunk_size=500,
            chunk_overlap=50
        )

        # Check if documents need to be indexed
        if pipeline.vector_store.get_document_count() == 0:
            pipeline.index_documents()
        else:
            pipeline.is_indexed = True

        return "✓ Pipeline initialized successfully!"
    except Exception as e:
        return f"✗ Error initializing pipeline: {str(e)}"


def reindex_documents():
    """Reindex all documents in the data folder."""
    global pipeline
    try:
        if pipeline is None:
            return "✗ Please initialize the pipeline first!"

        pipeline.index_documents(force_reindex=True)
        stats = pipeline.get_stats()
        return f"✓ Reindexing complete! Indexed {stats['document_count']} document chunks."
    except Exception as e:
        return f"✗ Error reindexing: {str(e)}"


def query_documents(query, top_k):
    """Query the RAG pipeline and return results."""
    global pipeline

    if pipeline is None:
        return "Please initialize the pipeline first!", ""

    if not query.strip():
        return "Please enter a query.", ""

    try:
        # Get retrieved documents
        results = pipeline.query(query, top_k=int(top_k))

        if not results:
            return "No relevant documents found.", ""

        # Format results for display
        output = "## Retrieved Documents\n\n"
        for i, result in enumerate(results, 1):
            filename = result['metadata'].get('filename', 'Unknown')
            chunk_id = result['metadata'].get('chunk_id', '?')
            distance = result.get('distance', 0)
            similarity = 1 - distance if distance is not None else 1.0

            output += f"### Document {i}: {filename} (Chunk {chunk_id})\n"
            output += f"**Similarity Score:** {similarity:.3f}\n\n"
            output += f"{result['content']}\n\n"
            output += "---\n\n"

        # Generate context for potential LLM use
        context = pipeline.generate_context(query, top_k=int(top_k))

        return output, context

    except Exception as e:
        return f"Error during query: {str(e)}", ""


def get_statistics():
    """Get pipeline statistics."""
    global pipeline

    if pipeline is None:
        return "Pipeline not initialized."

    try:
        stats = pipeline.get_stats()
        output = f"""
## Pipeline Statistics

- **Document Chunks:** {stats['document_count']}
- **Embedding Model:** {stats['model_name']}
- **Embedding Dimension:** {stats['embedding_dimension']}
- **Status:** {'Indexed ✓' if stats['is_indexed'] else 'Not Indexed'}
        """
        return output
    except Exception as e:
        return f"Error getting statistics: {str(e)}"


# Create Gradio Interface
with gr.Blocks(title="RAG Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 RAG Pipeline - Document Question Answering

    This application uses Retrieval Augmented Generation (RAG) to answer questions based on documents in the data folder.

    **Features:**
    - Sentence Transformer embeddings (Hugging Face)
    - ChromaDB vector database
    - Interactive document retrieval
    """)

    with gr.Tabs():
        # Query Tab
        with gr.Tab("Query Documents"):
            gr.Markdown("### Search for relevant information in your documents")

            with gr.Row():
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="e.g., What is machine learning?",
                    lines=2
                )

            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of documents to retrieve"
                )

            query_button = gr.Button("Search", variant="primary")

            with gr.Row():
                with gr.Column():
                    results_output = gr.Markdown(label="Retrieved Documents")

                with gr.Column():
                    context_output = gr.Textbox(
                        label="Context for LLM (Copy this to use with ChatGPT/Claude)",
                        lines=15,
                        max_lines=20
                    )

            query_button.click(
                fn=query_documents,
                inputs=[query_input, top_k_slider],
                outputs=[results_output, context_output]
            )

            # Example queries
            gr.Examples(
                examples=[
                    ["What is machine learning?", 3],
                    ["Explain neural networks", 3],
                    ["What are the applications of NLP?", 3],
                    ["How does backpropagation work?", 2],
                ],
                inputs=[query_input, top_k_slider]
            )

        # Management Tab
        with gr.Tab("Manage Pipeline"):
            gr.Markdown("### Initialize and manage the RAG pipeline")

            with gr.Row():
                init_button = gr.Button("Initialize Pipeline", variant="primary")
                reindex_button = gr.Button("Reindex Documents", variant="secondary")
                stats_button = gr.Button("Show Statistics")

            status_output = gr.Markdown()

            init_button.click(
                fn=initialize_pipeline,
                outputs=status_output
            )

            reindex_button.click(
                fn=reindex_documents,
                outputs=status_output
            )

            stats_button.click(
                fn=get_statistics,
                outputs=status_output
            )

            gr.Markdown("""
            ### How to use:

            1. **Initialize Pipeline**: Click to set up the RAG system and index documents
            2. **Reindex Documents**: Use this after adding new documents to the data folder
            3. **Show Statistics**: View information about indexed documents

            ### Adding new documents:

            - Place `.txt` or `.md` files in the `data/` folder
            - Click "Reindex Documents" to process new files
            """)

        # Info Tab
        with gr.Tab("Information"):
            gr.Markdown("""
            ## About this RAG Pipeline

            ### Components:

            1. **Document Loader**: Reads text files from the data folder
            2. **Text Chunker**: Splits documents into overlapping chunks for better retrieval
            3. **Sentence Transformer**: Generates embeddings using Hugging Face models
            4. **Vector Database**: ChromaDB for efficient similarity search
            5. **Gradio UI**: Interactive interface for querying documents

            ### Technical Details:

            - **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
            - **Chunk Size**: 500 characters with 50 character overlap
            - **Similarity Metric**: Cosine similarity
            - **Vector DB**: ChromaDB with persistent storage

            ### Use Cases:

            - Document question answering
            - Knowledge base search
            - Information retrieval
            - Context generation for LLMs

            ### Tips:

            - Ask specific questions for better results
            - Increase the number of retrieved documents for broader context
            - Use the generated context with external LLMs for enhanced answers
            """)


if __name__ == "__main__":
    # Auto-initialize on startup
    print("Starting RAG Pipeline Application...")
    init_message = initialize_pipeline()
    print(init_message)

    # Launch Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
