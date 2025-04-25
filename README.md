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

## 🧠 Example

```bash
PDF: artificial_intelligence_tutorial.pdf
Query: "Types of Intelligence?"
