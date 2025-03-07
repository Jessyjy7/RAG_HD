#!/usr/bin/env python3

import argparse
import torch
import numpy as np

# External libraries
import faiss  # CPU or GPU Faiss
from cuml.cluster import KMeans  # GPU-accelerated KMeans
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Local or custom utilities
from hadamardHD import kronecker_hadamard

def main():
    parser = argparse.ArgumentParser(description="Cluster HD with GPU KMeans (cuML) and optional index rebuild.")
    parser.add_argument("--dataset_name", type=str, default="neural-bridge/rag-dataset-12000",
                        help="Hugging Face dataset name/path to load.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Hugging Face embedding model name.")
    parser.add_argument("--index_path", type=str, default="faiss_index",
                        help="Path to save or load the FAISS index.")
    parser.add_argument("--D", type=int, default=65536,
                        help="Dimensionality for hypervector encoding.")
    parser.add_argument("--num_clusters", type=int, default=200,
                        help="Number of KMeans clusters.")
    parser.add_argument("--top_n_clusters", type=int, default=10,
                        help="Number of top clusters to consider during retrieval.")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of documents to retrieve.")
    parser.add_argument("--query", type=str, default="What is the role of neural networks in machine learning?",
                        help="Query string to retrieve documents for.")
    parser.add_argument("--build_index", action="store_true",
                        help="Whether to build a new FAISS index from the dataset.")
    parser.add_argument("--group_size", type=int, default=48,
                        help="Number of documents per bundling group.")
    args = parser.parse_args()

    gpu_available = torch.cuda.is_available()
    num_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0

    if gpu_available and num_gpus > 0:
        print(f"CUDA is available and Faiss detects {num_gpus} GPU(s).")
    else:
        print("Warning: CUDA or Faiss GPU not detected, proceeding on CPU.")

    print(f"\nLoading dataset '{args.dataset_name}' from Hugging Face...")
    dataset = load_dataset(args.dataset_name)["train"]
    documents = dataset["context"]
    num_docs = len(documents)
    print(f"Loaded {num_docs} documents.")

    print(f"\nInitializing Hugging Face embedding model '{args.model_name}'...")
    embedding_model = HuggingFaceEmbeddings(model_name=args.model_name)

    if args.build_index:
        print("\nBuilding new FAISS index...")
        vectorstore = FAISS.from_texts(texts=documents, embedding=embedding_model)
        vectorstore.save_local(args.index_path)
        print(f"Index saved to '{args.index_path}'")
    else:
        print(f"\nLoading existing FAISS index from '{args.index_path}'...")
        vectorstore = FAISS.load_local(args.index_path, embedding_model)

    index = vectorstore.index
    stored_vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
    print(f"Shape of stored_vectors: {stored_vectors.shape}")

    print(f"\nEncoding {stored_vectors.shape[0]} vectors to {args.D}D hypervectors...")
    base_matrix = np.random.uniform(-1, 1, (args.D, stored_vectors.shape[1]))
    base_matrix = np.where(base_matrix >= 0, 1, -1)

    def encode_to_hdc(vectors, base_mat):
        return np.matmul(base_mat, vectors.T).T

    hdc_vectors = encode_to_hdc(stored_vectors, base_matrix).astype(np.float32)
    print(f"HDC-encoded vectors shape: {hdc_vectors.shape}")

    print(f"\nRunning GPU-based KMeans (cuML) on {hdc_vectors.shape[0]} vectors with {args.num_clusters} clusters...")
    hdc_kmeans = KMeans(n_clusters=args.num_clusters, max_iter=300, init_method='scalable-kmeans++')
    cluster_assignments = hdc_kmeans.fit_predict(hdc_vectors)
    print("HDC clustering completed on GPU!")

    clustered_indices = [[] for _ in range(args.num_clusters)]
    for idx, cluster_id in enumerate(cluster_assignments):
        clustered_indices[cluster_id].append(idx)

    def bundle_cluster_docs_hadamard(cluster_doc_indices, hdc_vecs, hv_dim, group_size):
        cluster_doc_indices = sorted(cluster_doc_indices)
        bundled_hvs, bundle_doc_groups = [], []

        for start in range(0, len(cluster_doc_indices), group_size):
            group = cluster_doc_indices[start:start + group_size]
            if not group:
                break

            sum_binded = np.zeros(hv_dim)
            for doc_idx in group:
                key_vec = kronecker_hadamard(hv_dim, doc_idx)
                binded_HV = key_vec * hdc_vecs[doc_idx]
                sum_binded += binded_HV

            bundled_hvs.append(sum_binded)
            bundle_doc_groups.append(group)

        return bundled_hvs, bundle_doc_groups

    cluster_bundles, cluster_bundles_docs = [], []
    for cluster_docs in clustered_indices:
        if not cluster_docs:
            cluster_bundles.append([])
            cluster_bundles_docs.append([])
            continue
        bundled_hvs, doc_groups = bundle_cluster_docs_hadamard(cluster_docs, hdc_vectors, args.D, args.group_size)
        cluster_bundles.append(bundled_hvs)
        cluster_bundles_docs.append(doc_groups)

    print(f"\nHadamard Binding + Bundling done (group size={args.group_size})")

    def retrieve_from_hdc_hadamard(query_text, top_n_clusters, k):
        print(f"\nQuery: {query_text}")
        query_vec_384 = embedding_model.embed_query(query_text)
        query_hv = np.matmul(base_matrix, query_vec_384).astype(np.float32)

        cluster_centroids = hdc_kmeans.cluster_centers_
        distances = np.linalg.norm(cluster_centroids - query_hv, axis=1)
        top_clusters = np.argsort(distances)[:top_n_clusters]

        candidate_docs, candidate_dists = [], []
        for cluster in top_clusters:
            for bundle, doc_group in zip(cluster_bundles[cluster], cluster_bundles_docs[cluster]):
                for doc_idx in doc_group:
                    key_vec = kronecker_hadamard(args.D, doc_idx)
                    unboundHV = bundle * (1.0 / key_vec)
                    dist = np.linalg.norm(unboundHV - query_hv)

                    candidate_docs.append(doc_idx)
                    candidate_dists.append(dist)

        sorted_docs = np.argsort(candidate_dists)[:k]
        top_k_indices = np.array(candidate_docs)[sorted_docs]
        top_k_contexts = [documents[i] for i in top_k_indices]

        print("Retrieved document indices:", top_k_indices.tolist())
        return top_k_contexts, top_k_indices

    print("\n===== Retrieval Demo =====")
    retrieve_from_hdc_hadamard(args.query, args.top_n_clusters, args.k)

if __name__ == "__main__":
    main()

