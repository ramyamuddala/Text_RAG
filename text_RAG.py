import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and process PDF document
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Text chunking
def chunk_text(text, chunk_size=500, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    print(f"[INFO] Split into {len(chunks)} chunks")
    return chunks


# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def index_document(chunks):
    if not chunks:
        raise ValueError("[ERROR] No chunks to index. Check text loading and chunking.")
    
    chunk_embeddings = embedding_model.encode(chunks)
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    if chunk_embeddings.shape[0] == 0:
        raise ValueError("[ERROR] Embeddings not generated. Check SentenceTransformer model.")
    
    index.add(np.array(chunk_embeddings))
    print(f"[INFO] Indexed {len(chunks)} chunks")
    return index, chunk_embeddings


# Retrieve relevant chunks
def retrieve_top_k_chunks(query, index, chunks, k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)

    if len(indices[0]) == 0:
        raise ValueError("[ERROR] No relevant chunks retrieved. Try increasing 'k' or improving chunking.")

    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    print(f"[INFO] Retrieved {len(retrieved_chunks)} chunks")
    return retrieved_chunks


# Load pre-trained LLM for answer generation
qa_pipeline = pipeline("text-generation", model="openai-community/gpt2")

def generate_answer(query, index, chunks):
    retrieved_context = " ".join(retrieve_top_k_chunks(query, index, chunks)[:2])  # Use top 2 chunks
    input_text = f"Context: {retrieved_context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = qa_pipeline(input_text, max_new_tokens=150)[0]["generated_text"]
    return response


# RAG Pipeline
def rag_pipeline(pdf_path, query):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    index, _ = index_document(chunks)
    answer = generate_answer(query, index, chunks)
    return answer

# Example usage
if __name__ == "__main__":
    pdf_path = "artificial_intelligence_tutorial.pdf"  # Replace with your PDF
    query = "Types of Intelligence?"
    print("\n[AI Answer]:", rag_pipeline(pdf_path, query))
