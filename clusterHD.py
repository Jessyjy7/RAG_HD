#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import faiss  # CPU or GPU Faiss
from cuml.cluster import KMeans  # cuML for GPU KMeans
from datasets import load_dataset

# Updated LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Local utility (ensure you have this file in the same folder)
from hadamardHD import kronecker_hadamard


def main():
    parser = argparse.ArgumentParser(description="Cluster HD with GPU KMeans.")
    parser.add_argument("--dataset_name", type=str, default="neural-bridge/rag-dataset-12000")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--index_path", type=str, default="faiss_index")
    parser.add_argument("--D", type=int, default=65536)
    parser.add_argument("--num_clusters", type=int, default=200)
    parser.add_argument("--top_n_clusters", type=int, default=10)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--query", type=str, default="What is the role of neural networks in machine learning?")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--group_size", type=int, default=48)

    args = parser.parse_args()

    gpu_available = torch.cuda.is_available()
    print(f"CUDA available: {gpu_available}")
    num_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    print(f"FAISS GPU count: {num_gpus}")

    if gpu_available and num_gpus > 0:
        print("Using GPU-based FAISS.")
    else:
        print("Using CPU-based FAISS.")

    print(f"\nLoading dataset '{args.dataset_name}'...")
    dataset = load_dataset(args.dataset_name)["train"]
    documents = dataset["context"]

    print(f"\nInitializing embedding model '{args.model_name}'...")
    embedding_model = HuggingFaceEmbeddings(model_name=args.model_name)

    if args.build_index:
        print("Building new FAISS index...")
        vectorstore = FAISS.from_texts(texts=documents, embedding=embedding_model)
        vectorstore.save_local(args.index_path)
    else:
        print(f"Loading FAISS index from '{args.index_path}'...")
        vectorstore = FAISS.load_local(args.index_path, embedding_model)

    index = vectorstore.index
    stored_vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
    print(f"Stored vector shape: {stored_vectors.shape}")

    print(f"\nEncoding to {args.D}D hypervectors...")
    base_matrix = np.random.uniform(-1, 1, (args.D, stored_vectors.shape[1]))
    base_matrix = np.where(base_matrix >= 0, 1, -1)
    hdc_vectors = np.matmul(base_matrix, stored_vectors.T).T

    print(f"\nRunning cuML GPU KMeans clustering into {args.num_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.num_clusters, n_init=10, max_iter=300, output_type='numpy')
    cluster_assignments = kmeans.fit_predict(hdc_vectors)

    clustered_indices = [[] for _ in range(args.num_clusters)]
    for idx, cluster in enumerate(cluster_assignments):
        clustered_indices[cluster].append(idx)

    print("Clustering complete!")

    def bundle_cluster_docs_hadamard(cluster_doc_indices):
        bundled_hvs = []
        bundle_doc_groups = []
        cluster_doc_indices = sorted(cluster_doc_indices)

        for start in range(0, len(cluster_doc_indices), args.group_size):
            docs_slice = cluster_doc_indices[start:start + args.group_size]
            if not docs_slice:
                continue

            sum_binded = np.zeros(args.D)
            for doc_idx in docs_slice:
                key_vec = kronecker_hadamard(args.D, doc_idx)
                binded_HV = key_vec * hdc_vectors[doc_idx]
                sum_binded += binded_HV

            bundled_hvs.append(sum_binded)
            bundle_doc_groups.append(docs_slice)

        return bundled_hvs, bundle_doc_groups

    cluster_bundles = [[] for _ in range(args.num_clusters)]
    cluster_bundles_docs = [[] for _ in range(args.num_clusters)]

    for c_idx in range(args.num_clusters):
        bundled_hvs, bundled_docs = bundle_cluster_docs_hadamard(clustered_indices[c_idx])
        cluster_bundles[c_idx] = bundled_hvs
        cluster_bundles_docs[c_idx] = bundled_docs

    print(f"Bundling complete (group_size={args.group_size})!")

    def retrieve_from_hdc_hadamard(query_text):
        print(f"\nQuery: {query_text}")
        query_vec = embedding_model.embed_query(query_text)
        query_hv = np.matmul(base_matrix, query_vec)

        centroids = kmeans.cluster_centers_
        distances = np.linalg.norm(centroids - query_hv, axis=1)
        top_clusters = np.argsort(distances)[:args.top_n_clusters]
        print(f"Closest {args.top_n_clusters} clusters:", top_clusters.tolist())

        candidates = []
        distances = []

        for cluster_idx in top_clusters:
            for bundled_hv, doc_indices in zip(cluster_bundles[cluster_idx], cluster_bundles_docs[cluster_idx]):
                for doc_idx in doc_indices:
                    key_vec = kronecker_hadamard(args.D, doc_idx)
                    unbound_hv = bundled_hv * (1 / key_vec)
                    dist = np.linalg.norm(unbound_hv - query_hv)
                    candidates.append((doc_idx, dist))

        candidates = sorted(candidates, key=lambda x: x[1])[:args.k]
        top_indices = [idx for idx, _ in candidates]
        top_contexts = [documents[idx] for idx in top_indices]

        return top_contexts, top_indices

    print("\n===== Test Query Retrieval =====")
    top_k_contexts, top_k_indices = retrieve_from_hdc_hadamard(args.query)

    print("\nTop-k retrieved document indices:", top_k_indices)

if __name__ == "__main__":
    main()


