import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Embedding config
PROCESSED_DATA_PATH = "../data/processed/fastapi_clean.txt"
FAISS_INDEX_PATH = "../embeddings/faiss_index.faiss"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

os.makedirs("../embeddings", exist_ok=True)


# Load text
def load_clean_text():
    with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as r:
        return r.read()


# Chunk text
def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start::end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


# Generate embeddings
def gen_embeddings(chunks, model_name=EMBEDDING_MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


# Store embeddings in FAISS
def build_faiss_index(embeddings, path=FAISS_INDEX_PATH):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, path)
    print("faiss index saved")
    return index


# Main func
if __name__ == "__main__":
    text = load_clean_text()
    chunks = chunk_text(text)
    embeddings = gen_embeddings(chunks)
    build_faiss_index(np.array(embeddings))
