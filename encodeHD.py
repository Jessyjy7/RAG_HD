import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset
from sklearn.cluster import KMeans

#####################################################
# Step 1: Load Dataset & FAISS Vectors
#####################################################

print("Loading dataset from Hugging Face...")
dataset = load_dataset("neural-bridge/rag-dataset-12000")["train"]
documents = dataset["context"]  # The text field is 'context'
print(f"Loaded {len(documents)} documents.")

# Load FAISS index that has your document embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index",         # <-- Adjust if your FAISS index is saved elsewhere
    embedding_model,
    allow_dangerous_deserialization=True
)

index = vectorstore.index
num_vectors = index.ntotal
print(f"Total vectors stored in FAISS: {num_vectors}")

# Extract stored 384D embeddings from FAISS
print("Extracting vectors from FAISS for HDC encoding...")
stored_vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])
print(f"Shape of stored_vectors: {stored_vectors.shape}")


#####################################################
# Step 2: Hyperdimensional Encoding (HDC)
#####################################################

D = 16384  # Hypervector dimensionality
print(f"Encoding {num_vectors} vectors from 384D --> {D}D hypervectors...")

# Create a binary base matrix (D x 384) for mapping
base_matrix = np.random.uniform(-1, 1, (D, 384))
base_matrix = np.where(base_matrix >= 0, 1, -1)  # Convert base matrix to binary (-1,1)

def encode_to_hdc(vectors, base_matrix):
    """
    Encodes 384D FAISS vectors into D-dimensional hypervectors using matrix multiplication.
    This preserves real-valued output (no binarization after multiplication).
    """
    # (D x 384) dot (num_vectors x 384)^T => (D x num_vectors)^T => (num_vectors x D)
    enc_hvs = np.matmul(base_matrix, vectors.T).T
    return enc_hvs

hdc_vectors = encode_to_hdc(stored_vectors, base_matrix)
print(f"HDC-encoded vectors shape: {hdc_vectors.shape}")


#####################################################
# Step 3: K-Means Clustering in HDC Space
#####################################################

num_clusters = 100
print(f"Clustering {num_vectors} HDC hypervectors into {num_clusters} clusters...")

hdc_kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=300)
hdc_cluster_assignments = hdc_kmeans.fit_predict(hdc_vectors)
print("HDC clustering completed!")

# We'll store the cluster assignments and the data for retrieval
clustered_indices = [[] for _ in range(num_clusters)]
for i, c in enumerate(hdc_cluster_assignments):
    clustered_indices[c].append(i)

print("Cluster assignment completed!")


#####################################################
# Step 4: Query Retrieval in HDC Space
#####################################################

def retrieve_from_hdc(query, base_matrix, k=100):
    """
    Encodes the query in HDC space, finds the best cluster,
    then retrieves top-k hypervectors within that cluster
    based on distance to the query's hypervector.
    """
    print(f"\nQuery: {query}")
    
    # Encode query to hyperdimensional space
    query_vector_384 = embedding_model.embed_query(query)
    query_hv = np.matmul(base_matrix, query_vector_384)
    
    # 1) Find the nearest cluster centroid
    cluster_centroids = hdc_kmeans.cluster_centers_
    # Compare query_hv to each centroid
    distances_to_centroids = np.linalg.norm(cluster_centroids - query_hv, axis=1)
    best_cluster_idx = np.argmin(distances_to_centroids)
    
    print(f"Closest cluster to query: Cluster {best_cluster_idx}")

    # 2) Retrieve all points in that cluster
    candidate_indices = clustered_indices[best_cluster_idx]
    candidate_hvs = hdc_vectors[candidate_indices]
    
    # 3) Compute distances between query_hv and each candidate HV
    distances = np.linalg.norm(candidate_hvs - query_hv, axis=1)
    # Sort by ascending distance
    local_sorted = np.argsort(distances)[:k]
    
    # 4) Build final results
    top_k_indices = [candidate_indices[i] for i in local_sorted]
    top_k_contexts = [documents[idx] for idx in top_k_indices]
    
    # Print each retrieved doc chunk with index
    # print(f"\nHDC Clustering Top {k} Retrieved Documents:")
    # for rank, (doc_idx, context) in enumerate(zip(top_k_indices, top_k_contexts), start=1):
    #     print(f"{rank}. [Index: {doc_idx}]\n{context}\n")
    
    # Print final list of retrieved indices
    print("List of Retrieved Indices:", top_k_indices)
    return top_k_contexts, top_k_indices


#####################################################
# Step 5: Test Query
#####################################################

test_query = "What is the role of neural networks in machine learning?"
retrieved_contexts, retrieved_indices = retrieve_from_hdc(test_query, base_matrix, k=100)
