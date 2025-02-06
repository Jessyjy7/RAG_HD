from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  


loader = TextLoader("/Users/jessyjy7/Desktop/state_of_the_union.txt")  
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("faiss_index")
print("FAISS index saved!")

# retriever = vectorstore.as_retriever()
# docs = retriever.invoke("What's the president say about Apple?")

# for doc in docs:
#     print(doc)


vectorstore = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True  # Required to load FAISS
)
print("FAISS index loaded successfully!")

index = vectorstore.index
print(f"Total vectors stored in FAISS: {index.ntotal}")




# Get the total number of vectors
num_vectors = index.ntotal
if num_vectors > 0:
    vector = index.reconstruct(0)  
    print("First Vector Representation:", vector)
    print("Vector Dimensions:", len(vector))
else:
    print("No vectors found in FAISS.")
