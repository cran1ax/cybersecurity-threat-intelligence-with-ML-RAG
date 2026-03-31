import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

# =================================================================================
# 🌟 QWEN 2.5 3B (GGUF) LOCAL MODEL INTEGRATION 🌟
# If you have the downloaded Qwen .gguf file, paste the full path here!
# Make sure to use forward slashes (/) or double backslashes (\\)
# Example: QWEN_MODEL_PATH = "D:/Models/qwen2.5-3b-instruct-q4_k_m.gguf"
# =================================================================================
QWEN_MODEL_PATH = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf" 

is_qwen = False
if os.path.exists(QWEN_MODEL_PATH) and QWEN_MODEL_PATH.endswith(".gguf"):
    try:
        from langchain_community.llms import LlamaCpp
        print(f"Loading Qwen model from: {QWEN_MODEL_PATH}")
        llm = LlamaCpp(
            model_path=QWEN_MODEL_PATH,
            temperature=0.3,
            max_tokens=256,
            n_ctx=2048, # 2048 context window
            verbose=False
        )
        is_qwen = True
    except ImportError:
        print("ERROR: You must run `pip install llama-cpp-python` to use GGUF models!")
        llm = None
else:
    # Fallback to Flan-T5 if Qwen is not configured or path is wrong
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=120
    )

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Safe LLM (no errors)
from transformers import pipeline

# ✅ MUST MATCH IMPORT NAME
def build_vector_store(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    return FAISS.from_documents(documents, embeddings)

# ✅ MUST MATCH IMPORT NAME
def ask_rag(vector_db, question: str):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    if is_qwen and llm is not None:
        # Qwen uses Instruct/ChatML structure natively
        prompt = f"""<|im_start|>system
You are a helpful cybersecurity assistant. Use the provided context to answer the question concisely.<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{question}<|im_end|>
<|im_start|>assistant
"""
        # Call llama-cpp wrapper
        result = llm.invoke(prompt)
        return result.strip()
    else:
        # Instruction tuned models (like Flan-T5) just need a clear command.
        prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        # text2text-generation models don't return the prompt
        result = llm(prompt)
        return result[0]["generated_text"].strip()