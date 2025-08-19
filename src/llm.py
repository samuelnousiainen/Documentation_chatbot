from gpt4all import GPT4All

# CONFIG
MODEL_NAME = "ggml-gpt4all-j-v1.3-groovy"


# Load model
def load_model(model_name=MODEL_NAME):
    model = GPT4All(model_name)
    print("model loaded")
    return model


# Generate answer
def gen_answer(question, retrieved_chunks, model=None, max_tokens=512):
    """
    :param question: user query
    :param retrieved_chunks: (list[str]): Top-k relevant chunks from FAISS
    :param model: preloaded model instance
    :param max_tokens: max tokens for generation
    :return: str: Generated answer
    """
    if model is None:
        model = load_model()


    context = "\n".join(retrieved_chunks)

    # Prompt for LLM
    prompt = (
        f"You are an expert FastAPI assistant. "
        f"Answer the question using only the provided documentation context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    with model.chat_session():
        response = model.generate(prompt, max_tokens=max_tokens)

    return response