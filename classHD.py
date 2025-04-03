#!/usr/bin/env python3
"""
HDC Classification for Hugging Face RAG Datastore

This script encodes your Hugging Face datastore into hyperdimensional (HDC)
vectors and then uses a classification approach. We first perform clustering
(via Faiss) to assign each document a pseudo-label (i.e. class). We then compute a
centroid for each class and, at query time, encode the query into HDC space,
classify it to the nearest centroid, and finally rank documents only from that class.
"""

import argparse
import torch
import numpy as np
import faiss  
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings

# Assuming your custom Hadamard binding function is available
from hadamardHD import kronecker_hadamard


def main():
    parser = argparse.ArgumentParser(
        description="HDC Classification for Hugging Face RAG Datastore"
    )
    parser.add_argument("--dataset_name", type=str, default="neural-bridge/rag-dataset-12000",
                        help="Hugging Face dataset name/path to load.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Hugging Face embedding model name.")
    parser.add_argument("--index_path", type=str, default="faiss_index",
                        help="Path to save or load the FAISS index.")
    parser.add_argument("--D", type=int, default=8192,
                        help="Dimensionality for hypervector encoding.")
    parser.add_argument("--num_clusters", type=int, default=200,
                        help="Number of clusters (i.e. classes) for Faiss clustering.")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of documents to retrieve from the predicted class.")
    parser.add_argument("--query", type=str, default="What is the role of neural networks in machine learning?",
                        help="Query to retrieve documents for.")
    parser.add_argument("--build_index", action="store_true",
                        help="Whether to build a new FAISS index from the dataset.")
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

    # 4. Build or load the FAISS index for text retrieval (for the embedding model)
    if args.build_index:
        print("\nBuilding a new FAISS index (langchain_community)...")
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.from_texts(texts=documents, embedding=embedding_model)
        vectorstore.save_local(args.index_path)
        print(f"Index saved to '{args.index_path}'.")
    else:
        print(f"\nLoading existing FAISS index from '{args.index_path}'...")
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.load_local(
            args.index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    index = vectorstore.index
    num_vectors = index.ntotal
    print(f"Total vectors stored in FAISS: {num_vectors}")

    # 5. Reconstruct stored vectors (384D embeddings)
    stored_vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])
    print(f"Shape of stored_vectors: {stored_vectors.shape}")

    # 6. Hyperdimensional Encoding (HDC)
    D = args.D
    print(f"\nEncoding {num_vectors} vectors from 384D --> {D}D hypervectors...")

    # Create a random base matrix for encoding (values in {-1,1})
    base_matrix = np.random.uniform(-1, 1, (D, 384))
    base_matrix = np.where(base_matrix >= 0, 1, -1)

    def encode_to_hdc(vectors, base_mat):
        """
        Encodes 384D vectors into D-dimensional hypervectors.
        (D x 384) @ (num_vectors x 384)^T => (num_vectors x D)
        """
        return np.matmul(base_mat, vectors.T).T

    hdc_vectors = encode_to_hdc(stored_vectors, base_matrix)
    print(f"HDC-encoded vectors shape: {hdc_vectors.shape}")

    # 7. Clustering in HDC Space using Faiss (to get pseudo-labels for classification)
    print(f"\nClustering {num_vectors} hypervectors into {args.num_clusters} clusters with Faiss...")
    hdc_vectors = hdc_vectors.astype(np.float32)
    d = hdc_vectors.shape[1]
    ncentroids = args.num_clusters

    # Set up Faiss clustering
    clustering = faiss.Clustering(d, ncentroids)
    clustering.verbose = True
    clustering.niter = 300

    # Create a flat L2 index and move it to GPU if available
    index_flat = faiss.IndexFlatL2(d)
    if gpu_available and num_gpus > 0:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        index_used = gpu_index
    else:
        index_used = index_flat

    # Ensure hdc_vectors is C-contiguous
    hdc_vectors = np.ascontiguousarray(hdc_vectors)

    # Train clustering on the HDC vectors
    clustering.train(hdc_vectors, index_used)
    # Retrieve centroids from clustering (these are in HDC space)
    centroids_ptr = clustering.centroids
    cluster_centroids = np.ascontiguousarray(faiss.vector_float_to_array(centroids_ptr)).reshape(ncentroids, d)
    print("HDC clustering completed using Faiss!")

    # After clustering, assign each vector to its nearest centroid
    _, cluster_assignments = index_used.search(hdc_vectors, 1)
    cluster_assignments = cluster_assignments.flatten()

    # Group document indices by cluster (each cluster is a pseudo-class)
    clustered_indices = [[] for _ in range(ncentroids)]
    for i, c in enumerate(cluster_assignments):
        clustered_indices[c].append(i)
    print("Cluster assignment done!")

    # 8. Compute class centroids from HDC vectors (alternative to bundling)
    def compute_class_centroids(hdc_vecs, assignments, num_classes):
        centroids = np.zeros((num_classes, hdc_vecs.shape[1]), dtype=np.float32)
        counts = np.zeros(num_classes, dtype=np.int32)
        for idx, cl in enumerate(assignments):
            centroids[cl] += hdc_vecs[idx]
            counts[cl] += 1
        for cl in range(num_classes):
            if counts[cl] > 0:
                centroids[cl] /= counts[cl]
            else:
                centroids[cl] = np.zeros(hdc_vecs.shape[1])
        return centroids

    class_centroids = compute_class_centroids(hdc_vectors, cluster_assignments, ncentroids)
    print("Computed class centroids for classification.")

    # 9. Query Retrieval using Classification
    def retrieve_with_classification(query_text, embedding_model, base_matrix, 
                                     class_centroids, hdc_vecs, clustered_indices, documents, k=100):
        """
        1) Encode the query to 384D and then to HDC space.
        2) Compute the distance to each class centroid.
        3) Predict the class (cluster) with the nearest centroid.
        4) Within that class, rank documents by similarity to the query hypervector.
        """
        print(f"\nQuery: {query_text}")
        # Encode query to 384D vector then to HDC
        query_vec_384 = embedding_model.embed_query(query_text)
        query_hv = np.matmul(base_matrix, query_vec_384)
        
        # Compute L2 distances to each class centroid
        distances = np.linalg.norm(class_centroids - query_hv, axis=1)
        predicted_class = np.argmin(distances)
        print("Predicted class (cluster):", predicted_class)
        
        # Retrieve all documents in the predicted class
        doc_indices = clustered_indices[predicted_class]
        if not doc_indices:
            print("No documents found for this class!")
            return [], np.array([])
        
        # Rank the documents in the predicted class by distance to query_hv
        doc_dists = []
        for doc_idx in doc_indices:
            dist = np.linalg.norm(hdc_vecs[doc_idx] - query_hv)
            doc_dists.append(dist)
        doc_dists = np.array(doc_dists)
        sorted_order = np.argsort(doc_dists)[:k]
        top_doc_indices = np.array(doc_indices)[sorted_order]
        top_contexts = [documents[idx] for idx in top_doc_indices]
        
        print("Documents in predicted class:", doc_indices)
        print("Top k indices:", top_doc_indices.tolist())
        return top_contexts, top_doc_indices

    # 10. Run classification-based query retrieval
    print("\n===== Test Query Retrieval (Classification) =====")
    top_k_contexts, top_k_indices = retrieve_with_classification(
        query_text=args.query,
        embedding_model=embedding_model,
        base_matrix=base_matrix,
        class_centroids=class_centroids,
        hdc_vecs=hdc_vectors,
        clustered_indices=clustered_indices,
        documents=documents,
        k=args.k
    )

    # (Optional) Compare with a ground-truth list if available
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

    def compare_results(ground_truth_list, predicted_list):
        ground_truth_set = set(ground_truth_list)
        predicted_set = set(predicted_list)
        intersection = ground_truth_set.intersection(predicted_set)
        recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0
        precision = len(intersection) / len(predicted_set) if predicted_set else 0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        print("\n=== Ground Truth vs. Classification Comparison ===")
        print(f"Ground Truth Count: {len(ground_truth_set)}")
        print(f"Predicted Count:    {len(predicted_set)}")
        print(f"Overlap Count:      {len(intersection)}")
        print(f"Recall:    {recall:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"F1 Score:  {f1:.3f}")

    compare_results(ground_truth_indices, top_k_indices)


if __name__ == "__main__":
    main()
