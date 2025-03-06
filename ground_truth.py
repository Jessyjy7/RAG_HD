import argparse
import torch

def main():
    """
    Build or load a FAISS index and query it, using LangChain embeddings.
    Usage example:
      python clusterHD.py \
        --dataset_name "neural-bridge/rag-dataset-12000" \
        --model_name "sentence-transformers/all-MiniLM-L6-v2" \
        --query "What is the role of neural networks in machine learning?" \
        --top_k 5 \
        --build_index
    """

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build/Load FAISS index and retrieve documents.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="neural-bridge/rag-dataset-12000",
        help="The Hugging Face dataset name (or path) to load."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face embedding model name."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is the role of neural networks in machine learning?",
        help="Query to retrieve documents for."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of documents to retrieve."
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="faiss_index",
        help="Path to save or load the FAISS index."
    )
    parser.add_argument(
        "--build_index",
        action="store_true",
        help="Whether to build a new FAISS index from the dataset."
    )
    args = parser.parse_args()

    # 2. Check CUDA/GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print("CUDA is available. If Faiss is compiled with GPU support, it will use the GPU.")
    else:
        print("CUDA is not available. Using CPU-based Faiss.")

    # 3. Import Faiss and confirm GPU vs. CPU
    try:
        import faiss
        num_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        if gpu_available and num_gpus > 0:
            print(f"Faiss reports {num_gpus} GPU(s). GPU-based Faiss should be used.")
        else:
            print("Using CPU-based Faiss.")
    except ImportError:
        print("Faiss is not installed or not found. Please install faiss-cpu or faiss-gpu.")
        return

    # 4. Import other necessary libraries (inside main to avoid overhead if imports fail)
    from datasets import load_dataset
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

    # 5. Load the dataset
    print(f"\nLoading dataset '{args.dataset_name}' from Hugging Face...")
    dataset = load_dataset(args.dataset_name)["train"]
    documents = dataset["context"]
    print(f"Loaded {len(documents)} documents.")

    # 6. Initialize embedding model
    print(f"\nGenerating embeddings using Hugging Face model '{args.model_name}'...")
    embedding_model = HuggingFaceEmbeddings(model_name=args.model_name)

    # 7. Build FAISS index if requested
    if args.build_index:
        print("\nBuilding FAISS index...")
        documents_with_metadata = [
            {"page_content": doc, "metadata": {"index": i}}
            for i, doc in enumerate(documents)
        ]

        vectorstore = FAISS.from_texts(
            [doc["page_content"] for doc in documents_with_metadata],
            embedding_model,
            metadatas=[doc["metadata"] for doc in documents_with_metadata],
        )
        vectorstore.save_local(args.index_path)
        print(f"FAISS index built and saved to '{args.index_path}'!")
    else:
        print(f"\nSkipping index build. Using existing index at '{args.index_path}'...")

    # 8. Load the FAISS index for retrieval
    print("\nLoading FAISS index for retrieval...")
    vectorstore = FAISS.load_local(
        args.index_path,
        embedding_model,
        allow_dangerous_deserialization=True  # Use with caution in production
    )

    # 9. Verify the number of vectors
    index = vectorstore.index
    num_vectors = index.ntotal
    print(f"Total vectors stored in FAISS: {num_vectors}")

    # 10. Define a retrieval function
    def retrieve_top_k(query_text, k=10):
        """Retrieve top-k documents and their indices using FAISS."""
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query_text)

        top_k_contexts = [doc.page_content for doc in retrieved_docs]
        top_k_indices = [doc.metadata.get("index", -1) for doc in retrieved_docs]

        print("\nList of Retrieved Indices:", top_k_indices)
        return top_k_contexts, top_k_indices

    # 11. Run retrieval with user-provided query
    print(f"\nRunning retrieval for query: '{args.query}'")
    retrieve_top_k(args.query, k=args.top_k)

if __name__ == "__main__":
    main()
