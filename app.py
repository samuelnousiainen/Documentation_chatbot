import streamlit as st
from src.retrieval import load_faiss, load_chunks, retrieve
from src.llm import load_model, gen_answer

# Page config

st.set_page_config(page_title="FastApi Documentation Bot")
st.title("FastApi Documentation Bot")
st.write("Ask questions about FastApi documentation")


# Load faiss and chunks
@st.cache_resource
def load_index_and_chunks():
    index = load_faiss()
    chunks = load_chunks()
    return index, chunks


index, chunks = load_index_and_chunks()


# Load model
@st.cache_resource
def load_llm():
    return load_model()


model = load_llm()


# Conversation history
if "history" not in st.session_state:
    st.session_state.history = []


# User input
query = st.text_input("Ask anything about FastApi")

if query:
    retrieved_chunks = retrieve(query, index, chunks)

    answer = gen_answer(query, retrieved_chunks, model=model)

    st.session_state.history.append({"query": query, "answer": answer})


# Display
for chat in st.session_state.history[::-1]:
    st.markdown(f"**You:**{chat[query]}")
    st.markdown(f"**Bot**{chat[answer]}")
    st.markdown("---")
