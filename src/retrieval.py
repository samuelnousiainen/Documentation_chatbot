import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Retrieval config
FAISS_INDEX_PATH = "../embeddings/faiss_index.faiss"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
PROCESSED_DATA_PATH = "../data/processed/fastapi_cleaned.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Load FAISS
def load_faiss(path=FAISS_INDEX_PATH):
    index = faiss.read_index(path)
    print("index loaded")
    return index


# Load chunks
def load_chunks(file_path=PROCESSED_DATA_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    with open(file_path, "r", encoding="utf-8") as r:
        text = r.read()
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - chunk_overlap
    return chunks


# Embed query
def embed_query(query, model_name=EMBEDDING_MODEL_NAME):
    model = SentenceTransformer(model_name)
    return model.encode([query])


# Retrieve top chunks
def retrieve(query, index, chunks, top_k=TOP_K):
    query_vector = embed_query(query)
    D, I = index.search(np.array(query_vector), top_k)
    retrieved = [chunks[i] for i in I[0]]
    return retrieved
