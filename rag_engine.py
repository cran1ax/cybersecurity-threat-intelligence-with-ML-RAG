from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Safe LLM (no errors)
from transformers import pipeline

llm = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=120,
    framework="pt"   # ✅ THIS FIXES EVERYTHING
)

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

    prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    result = llm(prompt)
    return result[0]["generated_text"].strip()