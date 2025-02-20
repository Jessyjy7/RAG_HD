import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset

print("Loading dataset from Hugging Face...")
dataset = load_dataset("neural-bridge/rag-dataset-12000")["train"]
documents = dataset["context"]  
print(f"Loaded {len(documents)} documents.")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(documents, embedding_model)
vectorstore.save_local("faiss_index_multi_cluster")
print("FAISS index created & saved!")

vectorstore = FAISS.load_local(
    "faiss_index_multi_cluster",
    embedding_model,
    allow_dangerous_deserialization=True
)

index = vectorstore.index
num_vectors = index.ntotal
print(f"Total vectors stored in FAISS: {num_vectors}")

stored_vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])
print("Extracted FAISS vectors for clustering...")

num_clusters = 200
kmeans = faiss.Kmeans(d=stored_vectors.shape[1], k=num_clusters, niter=50, nredo=5)
kmeans.train(stored_vectors)

_, cluster_assignments = kmeans.index.search(stored_vectors, 1)
print(f"Clustering completed with {num_clusters} clusters!")

def multi_cluster_retrieve(query, top_n_clusters=3, k=100):
    """Retrieve top-k documents by picking top-n nearest clusters, then re-ranking those docs by distance."""
    query_vector = np.array([embedding_model.embed_query(query)])
    
    cluster_centroids = kmeans.centroids  

    centroid_distances = np.linalg.norm(cluster_centroids - query_vector, axis=1)
    sorted_cluster_indices = np.argsort(centroid_distances)[:top_n_clusters]

    print(f"Top-{top_n_clusters} cluster indices: {sorted_cluster_indices}")
    
    candidate_indices = []
    for c_idx in sorted_cluster_indices:
        c_doc_indices = np.where(cluster_assignments == c_idx)[0]
        candidate_indices.extend(c_doc_indices)
    
    candidate_indices = list(set(candidate_indices))  

    candidate_vectors = stored_vectors[candidate_indices]
    distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)
    sorted_candidates = np.argsort(distances)[:k]
    
    top_k_indices = [candidate_indices[i] for i in sorted_candidates]
    top_k_contexts = [documents[idx] for idx in top_k_indices]

    # print(f"\nMulti-Cluster Retrieval (top-{top_n_clusters} clusters) - Top {k} docs:")
    # for rank, (doc_idx, context) in enumerate(zip(top_k_indices, top_k_contexts), start=1):
    #     print(f"{rank}. [Index: {doc_idx}]\n{context}\n")

    print("Final List of Retrieved Indices:", top_k_indices)
    return top_k_contexts, top_k_indices

# ---- Test Query ---- #
test_query = "What is the role of neural networks in machine learning?"
retrieved_contexts, retrieved_indices = multi_cluster_retrieve(test_query, top_n_clusters=3, k=100)

