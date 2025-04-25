# 🔍 PDF-Based Text Retrieval + Generation (RAG) Pipeline

This project lets you ask questions about the content of **any PDF**.  
It combines text extraction, chunking, semantic search, and generation.  
Built as a lightweight demo of how RAG works — simple, fast, and no black magic.

---

## ⚙️ What it Does

- Pulls text from a PDF using `pdfplumber`
- Splits the text into manageable overlapping chunks
- Embeds the chunks with `sentence-transformers`
- Indexes everything using `FAISS` for fast similarity search
- Retrieves the most relevant chunks for a query
- Sends those chunks + query to a GPT2-based text generation model
---
## 🛠️ Tech Stack
📄 PDF Text Extraction: pdfplumber

🧠 Embeddings: sentence-transformers (all-MiniLM-L6-v2)

🧠 Similarity Search: faiss

💬 Generation: transformers (gpt2)

🔧 Text Splitting: langchain.text_splitter

## 🧪 Notes & Warnings
GPT2 doesn’t have real-world knowledge past 2021 and isn’t great with factual accuracy. You can swap it with a better model if needed.

It doesn't support multiple PDFs (yet).

No web UI, no fast API — it’s just pure Python, keepin’ it real.

##🧱 Next Steps (What You Can Add)
Streamlit/Gradio interface

Support for multiple PDFs or folders

Cache for embeddings so you don’t recompute every time

Use a better generator model like LLaMA, Mistral, or GPT-Neo
---


## 🧠 Example

```bash
PDF: artificial_intelligence_tutorial.pdf
Query: "Types of Intelligence?"

##Output:

[AI Answer]: There are multiple types of intelligence such as natural, artificial, emotional, etc...

