import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset

print("Loading dataset from Hugging Face...")
dataset = load_dataset("neural-bridge/rag-dataset-12000")["train"]
documents = dataset["context"] 
print(f"Loaded {len(documents)} documents.")

print("Generating embeddings using Hugging Face model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents_with_metadata = [
    {"page_content": doc, "metadata": {"index": i}} for i, doc in enumerate(documents)
]

vectorstore = FAISS.from_texts(
    [doc["page_content"] for doc in documents_with_metadata], 
    embedding_model, 
    metadatas=[doc["metadata"] for doc in documents_with_metadata]
)

vectorstore.save_local("faiss_index")
print("FAISS index saved with metadata!")

print("Loading FAISS index for retrieval...")
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

index = vectorstore.index
num_vectors = index.ntotal
print(f"Total vectors stored in FAISS: {num_vectors}")

def retrieve_top_k(query, k=10):
    """Retrieve top-k documents and their indices using FAISS."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)

    top_k_contexts = [doc.page_content for doc in retrieved_docs]
    top_k_indices = [doc.metadata.get("index", -1) for doc in retrieved_docs]  # Use .get() to prevent KeyError

    # print("\n FAISS Ground Truth Top-K Retrieved Chunks:")
    # for idx, (context, doc_idx) in enumerate(zip(top_k_contexts, top_k_indices)):
    #     print(f"{idx+1}. [Index: {doc_idx}]\n{context}\n")

    print(" List of Retrieved Indices:", top_k_indices)
    return top_k_contexts, top_k_indices

# ---- Test Query ---- #
query = "What is the role of neural networks in machine learning?"
retrieved_gt_contexts, retrieved_gt_indices = retrieve_top_k(query, k=100)
