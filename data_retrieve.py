import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  
from hadamardHD import binding, bundling, unbundling


loader = TextLoader("/Users/jessyjy7/Desktop/state_of_the_union.txt")  
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(texts, embedding_model)
vectorstore.save_local("faiss_index")
print("FAISS index saved!")

vectorstore = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True  
)
print("FAISS index loaded successfully!")

index = vectorstore.index
num_vectors = index.ntotal
print(f"Total vectors stored in FAISS: {num_vectors}")

if num_vectors > 0:
    stored_vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])
    print("Extracted 384D Vectors Shape:", stored_vectors.shape)  
else:
    print("No vectors found in FAISS.")

D = 65536

base_matrix = np.random.uniform(-1, 1, (D, 384))
base_matrix = np.where(base_matrix >= 0, 1, -1)  
pseudo_inverse = np.linalg.pinv(base_matrix)  

def encode_faiss_vectors(stored_vectors, base_matrix, D):
    """
    Encodes 384D FAISS vectors into D-dimensional HDC hypervectors using random projection.
    The result remains **real-valued** and is NOT binarized.
    """
    enc_hvs = np.matmul(base_matrix, stored_vectors.T)  
    return enc_hvs.T  

encoded_hvs = encode_faiss_vectors(stored_vectors, base_matrix, D)

print(f"Encoded Hypervectors Shape: {encoded_hvs.shape}") 

def bundle_hypervectors(encoded_hvs, D):
    """Bundles encoded hypervectors using Hadamard binding."""
    bundled_HV = np.zeros(D)  
    for i, hv in enumerate(encoded_hvs):
        binded_HV = binding(D, i, hv)  
        bundled_HV += binded_HV  
    return bundled_HV

bundled_hv = bundle_hypervectors(encoded_hvs, D)

np.save("bundled_hv.npy", bundled_hv)
np.save("original_384D_vectors.npy", stored_vectors)

print("Bundled Hypervector Stored Successfully!")

