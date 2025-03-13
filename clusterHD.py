#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import faiss  
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from hadamardHD import kronecker_hadamard


def main():
    """
    clusterHD.py using GPU-accelerated Faiss for KMeans-like clustering:
      - Uses Faiss.Clustering with a GPU index when available.
      - Fixes centroids pointer via faiss.vector_float_to_array().
    """

    parser = argparse.ArgumentParser(description="Cluster HD with GPU Faiss Clustering & new LangChain community FAISS.")
    parser.add_argument("--dataset_name", type=str, default="neural-bridge/rag-dataset-12000",
                        help="Hugging Face dataset name/path to load.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Hugging Face embedding model name.")
    parser.add_argument("--index_path", type=str, default="faiss_index",
                        help="Path to save or load the FAISS index.")
    parser.add_argument("--D", type=int, default=16384,
                        help="Dimensionality for hypervector encoding.")
    parser.add_argument("--num_clusters", type=int, default=200,
                        help="Number of clusters for Faiss clustering.")
    parser.add_argument("--top_n_clusters", type=int, default=10,
                        help="Number of top clusters to consider during retrieval.")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of documents to retrieve.")
    parser.add_argument("--query", type=str, default="What is the role of neural networks in machine learning?",
                        help="Query to retrieve documents for.")
    parser.add_argument("--build_index", action="store_true",
                        help="Whether to build a new FAISS index from the dataset.")
    parser.add_argument("--group_size", type=int, default=48,
                        help="Number of documents per bundling group.")
    args = parser.parse_args()

    # 1. Check CUDA availability
    gpu_available = torch.cuda.is_available()
    print(f"CUDA available: {gpu_available}")

    num_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    print(f"FAISS GPU count: {num_gpus}")

    if gpu_available and num_gpus > 0:
        print("Using GPU-based Faiss for indexing and clustering.")
    else:
        print("Using CPU-based Faiss for indexing and clustering.")

    # 2. Load dataset
    print(f"\nLoading dataset '{args.dataset_name}' from Hugging Face...")
    dataset = load_dataset(args.dataset_name)["train"]
    documents = dataset["context"]
    print(f"Loaded {len(documents)} documents.")

    # 3. Load embedding model
    print(f"\nInitializing embedding model '{args.model_name}'...")
    embedding_model = HuggingFaceEmbeddings(model_name=args.model_name)

    # 4. Build or load the FAISS index for text retrieval
    if args.build_index:
        print("\nBuilding a new FAISS index (langchain_community)...")
        vectorstore = FAISS.from_texts(texts=documents, embedding=embedding_model)
        vectorstore.save_local(args.index_path)
        print(f"Index saved to '{args.index_path}'.")
    else:
        print(f"\nLoading existing FAISS index from '{args.index_path}'...")

    # Load with allow_dangerous_deserialization=True (new approach)
    vectorstore = FAISS.load_local(
        args.index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    index = vectorstore.index
    num_vectors = index.ntotal
    print(f"Total vectors stored in FAISS: {num_vectors}")

    # 5. Reconstruct stored vectors
    stored_vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])
    print(f"Shape of stored_vectors: {stored_vectors.shape}")

    # 6. Hyperdimensional Encoding (HDC)
    D = args.D
    print(f"\nEncoding {num_vectors} vectors from 384D --> {D}D hypervectors...")

    base_matrix = np.random.uniform(-1, 1, (D, 384))
    base_matrix = np.where(base_matrix >= 0, 1, -1)

    def encode_to_hdc(vectors, base_mat):
        """
        Encodes 384D vectors into D-dimensional hypervectors (real values).
        (D x 384) * (num_vectors x 384)^T => (num_vectors x D)
        """
        return np.matmul(base_mat, vectors.T).T

    hdc_vectors = encode_to_hdc(stored_vectors, base_matrix)
    print(f"HDC-encoded vectors shape: {hdc_vectors.shape}")

    # 7. GPU-Accelerated Clustering in HDC Space using Faiss
    print(f"\nClustering {num_vectors} hypervectors into {args.num_clusters} clusters with Faiss...")
    # Faiss requires float32 data
    hdc_vectors = hdc_vectors.astype(np.float32)
    d = hdc_vectors.shape[1]
    ncentroids = args.num_clusters

    # Set up Faiss clustering
    clustering = faiss.Clustering(d, ncentroids)
    clustering.verbose = True
    clustering.niter = 300

    # Create a flat (L2) index and move it to GPU if available
    index_flat = faiss.IndexFlatL2(d)
    if gpu_available and num_gpus > 0:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        index_used = gpu_index
    else:
        index_used = index_flat

    # Ensure hdc_vectors is C-contiguous
    hdc_vectors = np.ascontiguousarray(hdc_vectors)

    # Train the clustering on the HDC vectors
    clustering.train(hdc_vectors, index_used)

    centroids_ptr = clustering.centroids  # raw pointer to centroid data
    centroids = np.ascontiguousarray(faiss.vector_float_to_array(centroids_ptr)).reshape(ncentroids, d)

    # After training, assign each vector to its nearest centroid
    _, cluster_assignments = index_used.search(hdc_vectors, 1)
    cluster_assignments = cluster_assignments.flatten()
    print("HDC clustering completed using Faiss!")

    # Group doc indices by cluster
    clustered_indices = [[] for _ in range(ncentroids)]
    for i, c in enumerate(cluster_assignments):
        clustered_indices[c].append(i)

    print("Cluster assignment done!")

    # 8. Unique Hadamard Keys & Bundling
    def bundle_cluster_docs_hadamard(cluster_doc_indices, hdc_vecs, hv_dim, group_size):
        cluster_doc_indices = sorted(cluster_doc_indices)
        bundled_hvs = []
        bundle_doc_groups = []

        for start_idx in range(0, len(cluster_doc_indices), group_size):
            docs_slice = cluster_doc_indices[start_idx : start_idx + group_size]
            if not docs_slice:
                break

            sum_binded = np.zeros(hv_dim)
            for doc_idx in docs_slice:
                key_vec = kronecker_hadamard(hv_dim, doc_idx)
                docHV = hdc_vecs[doc_idx]
                binded_HV = key_vec * docHV
                sum_binded += binded_HV

            bundled_hvs.append(sum_binded)
            bundle_doc_groups.append(docs_slice)

        return bundled_hvs, bundle_doc_groups

    cluster_bundles = [[] for _ in range(ncentroids)]
    cluster_bundles_docs = [[] for _ in range(ncentroids)]

    for c_idx in range(ncentroids):
        doc_indices_c = clustered_indices[c_idx]
        if doc_indices_c:
            cluster_bundled_hvs, cluster_bundled_docs = bundle_cluster_docs_hadamard(
                doc_indices_c,
                hdc_vectors,
                D,
                args.group_size
            )
            cluster_bundles[c_idx] = cluster_bundled_hvs
            cluster_bundles_docs[c_idx] = cluster_bundled_docs

    print(f"Hadamard Binding + Bundling done! (group_size={args.group_size})")

    # 9. Query Retrieval with "Partial Unbundling"
    def retrieve_from_hdc_hadamard(query_text, top_n_clusters=20, k=100):
        """
        1) Encode query => query HV
        2) Find the TOP_n nearest cluster centroids
        3) Combine all doc bundles from those clusters
        4) "Unbundle" each doc => measure distance => pick top-k
        """
        print(f"\nQuery: {query_text}")

        # Convert query to 384D using the embedding model
        query_vec_384 = embedding_model.embed_query(query_text)
        # Map query to D-dim hypervector
        query_hv = np.matmul(base_matrix, query_vec_384)

        # Use the centroids from Faiss clustering
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

        print("Final List of Indices:", top_k_indices.tolist())
        return top_k_contexts, top_k_indices

    # 10. Run retrieval with user-provided query
    print("\n===== Test Query Retrieval =====")
    top_k_contexts, top_k_indices = retrieve_from_hdc_hadamard(
        query_text=args.query,
        top_n_clusters=args.top_n_clusters,
        k=args.k
    )

    # Example: Compare results with a ground-truth list
    ground_truth_indices = [
        5781, 9413, 7181, 3004, 5634, 7784, 8005, 2378, 464, 4915,
        4113, 6030, 2680, 8954, 6685, 3525, 2494, 7756, 2243, 409,
        7694, 1953, 8445, 176, 698, 5946, 9143, 6495, 8989, 3118,
        174, 7405, 59, 4251, 3783, 8795, 700, 3829, 6943, 3079,
        7893, 9295, 1921, 2569, 5988, 1061, 5546, 3095, 1126, 2809,
        2166, 3584, 7448, 1374, 190, 2336, 4976, 2011, 7777, 5639,
        7005, 1514, 7096, 1962, 1335, 2837, 3839, 7476, 8461, 768,
        1994, 4187, 9591, 1352, 1598, 9480, 8492, 7841, 8716, 5067,
        6484, 2950, 4757, 7500, 1862, 4890, 5235, 1357, 1119, 2977,
        4100, 6447, 7697, 3653, 8773, 7663, 273, 270, 3660, 2440
    ]

    def compare_results(ground_truth_list, clusterHD_list):
        ground_truth_set = set(ground_truth_list)
        clusterHD_set = set(clusterHD_list)
        intersection = ground_truth_set.intersection(clusterHD_set)
        recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0
        precision = len(intersection) / len(clusterHD_set) if clusterHD_set else 0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        print("\n=== Ground Truth vs. ClusterHD Comparison ===")
        print(f"Ground Truth Count: {len(ground_truth_set)}")
        print(f"ClusterHD Count:   {len(clusterHD_set)}")
        print(f"Overlap Count:     {len(intersection)}")
        print(f"Recall:    {recall:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"F1 Score:  {f1:.3f}")

    compare_results(ground_truth_indices, top_k_indices)


if __name__ == "__main__":
    main()

