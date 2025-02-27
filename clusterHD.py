import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset
from sklearn.cluster import KMeans

# Hadamard-based utilities
from hadamardHD import kronecker_hadamard

########################################################
# Step 1: Load Dataset & FAISS Vectors
########################################################

print("Loading dataset from Hugging Face...")
dataset = load_dataset("neural-bridge/rag-dataset-12000")["train"]
documents = dataset["context"]  
num_docs = len(documents)
print(f"Loaded {num_docs} documents.")

# Load existing FAISS index with 384D embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index",  
    embedding_model,
    allow_dangerous_deserialization=True
)
index = vectorstore.index
num_vectors = index.ntotal
print(f"Total vectors stored in FAISS: {num_vectors}")

stored_vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])
print(f"Shape of stored_vectors: {stored_vectors.shape}")

########################################################
# Step 2: Hyperdimensional Encoding (HDC)
########################################################

D = 16384  # Hypervector dimensionality
print(f"Encoding {num_vectors} vectors from 384D --> {D}D hypervectors...")

# Create a binary base matrix (D x 384) for mapping
base_matrix = np.random.uniform(-1, 1, (D, 384))
base_matrix = np.where(base_matrix >= 0, 1, -1)

def encode_to_hdc(vectors, base_matrix):
    """
    Encodes 384D vectors into D-dimensional hypervectors (real values).
    (D x 384) * (num_vectors x 384)^T => (num_vectors x D)
    """
    return np.matmul(base_matrix, vectors.T).T

hdc_vectors = encode_to_hdc(stored_vectors, base_matrix)
print(f"HDC-encoded vectors shape: {hdc_vectors.shape}")

########################################################
# Step 3: K-Means Clustering in HDC Space (200 Clusters)
########################################################

num_clusters = 200
print(f"Clustering {num_vectors} hypervectors into {num_clusters} clusters...")

hdc_kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=300)
cluster_assignments = hdc_kmeans.fit_predict(hdc_vectors)
print("HDC clustering completed!")

clustered_indices = [[] for _ in range(num_clusters)]
for i, c in enumerate(cluster_assignments):
    clustered_indices[c].append(i)

print("Cluster assignment done!")

########################################################
# Step 4: Unique Hadamard Keys & Bundling in Groups of 100
########################################################

def bundle_cluster_docs_hadamard(cluster_doc_indices, hdc_vectors, hv_dim):
    """
    For each cluster:
      - Sort doc indices,
      - Process docs in groups of 100,
      - Each doc i has a unique Hadamard key => row i in kronecker_hadamard(hv_dim, i).
      - Bind doc i's HV => key_i * HV_i,
      - Sum them => 1 "bundled HV" per group,
      - Track doc indices in that group.
    Returns:
      - A list of (bundled HV),
      - A parallel list of doc index groups
    """
    cluster_doc_indices = sorted(cluster_doc_indices)

    bundled_hvs = []
    bundle_doc_groups = []

    # Change here for bundle size
    for start_idx in range(0, len(cluster_doc_indices), 48):
        docs_slice = cluster_doc_indices[start_idx : start_idx + 48]
        if not docs_slice:
            break
        
        sum_binded = np.zeros(hv_dim)
        
        for doc_idx in docs_slice:
            key_vec = kronecker_hadamard(hv_dim, doc_idx)
            docHV = hdc_vectors[doc_idx]
            binded_HV = key_vec * docHV
            sum_binded += binded_HV

        bundled_hvs.append(sum_binded)
        bundle_doc_groups.append(docs_slice)
    
    return bundled_hvs, bundle_doc_groups

cluster_bundles = [[] for _ in range(num_clusters)]
cluster_bundles_docs = [[] for _ in range(num_clusters)]

for c_idx in range(num_clusters):
    doc_indices_c = clustered_indices[c_idx]
    if doc_indices_c:
        cluster_bundled_hvs, cluster_bundled_docs = bundle_cluster_docs_hadamard(
            doc_indices_c,
            hdc_vectors,
            D
        )
        cluster_bundles[c_idx] = cluster_bundled_hvs
        cluster_bundles_docs[c_idx] = cluster_bundled_docs

print("Hadamard Binding + Bundling in groups of 100 done!")

########################################################
# Step 5: Query Retrieval with "Partial Unbundling" (Top 3 Clusters)
########################################################

def retrieve_from_hdc_hadamard(query, top_n_clusters=20, k=100):
    """
    1) Encode query => query HV
    2) Find the TOP_n nearest cluster centroids
    3) Combine all doc bundles from those clusters
    4) "Unbundle" each doc => measure distance => pick top-k
    Using doc_idx as the unique hadamard key row index.
    """
    print(f"\nQuery: {query}")

    query_vec_384 = embedding_model.embed_query(query)
    query_hv = np.matmul(base_matrix, query_vec_384)

    centroids = hdc_kmeans.cluster_centers_
    dist_to_centroids = np.linalg.norm(centroids - query_hv, axis=1)
    sorted_cluster_indices = np.argsort(dist_to_centroids)[:top_n_clusters]
    print(f"Closest {top_n_clusters} clusters:", sorted_cluster_indices)

    candidate_doc_indices = []
    candidate_distances = []

    for best_cluster in sorted_cluster_indices:
        cluster_bundledHVs = cluster_bundles[best_cluster]
        cluster_bundledDocs = cluster_bundles_docs[best_cluster]

        for bundleHV, doc_indices_slice in zip(cluster_bundledHVs, cluster_bundledDocs):
            for doc_idx in doc_indices_slice:
                key_vec = kronecker_hadamard(D, doc_idx)
                unboundHV = bundleHV * (1.0 / key_vec)
                dist = np.linalg.norm(unboundHV - query_hv)

                candidate_doc_indices.append(doc_idx)
                candidate_distances.append(dist)

    candidate_doc_indices = np.array(candidate_doc_indices)
    candidate_distances = np.array(candidate_distances)
    sorted_candidates = np.argsort(candidate_distances)[:k]
    top_k_indices = candidate_doc_indices[sorted_candidates]
    top_k_contexts = [documents[idx] for idx in top_k_indices]

    # print(f"\nHDC Hadamard => Top {k} docs from top {top_n_clusters} clusters (bundled by 100):")
    # for rank, (doc_idx, context) in enumerate(zip(top_k_indices, top_k_contexts), start=1):
    #     print(f"{rank}. [Index: {doc_idx}]\n{context}\n")

    print("Final List of Indices:", top_k_indices.tolist())
    return top_k_contexts, top_k_indices

########################################################
# Step 6: Test Query
########################################################

test_query = "What is the role of neural networks in machine learning?"
top_k_contexts, top_k_ids = retrieve_from_hdc_hadamard(query=test_query, top_n_clusters=10, k=100)

