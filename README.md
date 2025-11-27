# 📘 RAG Document Query Tool using Gemini 2.5 Flash + ChromaDB

A production-style **Retrieval Augmented Generation (RAG)** system built using:

- **Gemini 2.5 Flash** (LLM generation)  
- **Gemini text-embedding-004** (embeddings)  
- **ChromaDB Persistent Vector Store**  
- **Pickle-based embedding caching**  
- **PDF/Text ingestion & chunking**  
- **Top-K retrieval**  
- **Grounding score**  
- **Hallucination detection**

---

## 🚀 Features

### ✅ Gemini 2.5 Flash for RAG Generation  
Uses the official `google-generativeai` SDK.

### ✅ text-embedding-004 Embeddings  
Lightweight and globally available.

### ✅ Persistent ChromaDB Vector Store  
Embeddings are saved locally inside `./vector_db/`.

### ✅ Embedding Pickle Cache  
- Embeddings are saved at `vector_db/embeddings.pkl`  
- On next run → embeddings load instantly  
- Saves API cost + time

### ✅ PDF Ingestion & Chunking  
Extracts text and splits into overlapping chunks.

### ✅ Retrieval (Top-K Similarity Search)  
Fast vector search powered by Chroma.

### ✅ Grounded RAG Prompting  
Uses retrieved context to generate grounded answers.

### ✅ Grounding Score + Hallucination Detection  
Cosine similarity between:  
- embeddings(answer)  
- embeddings(retrieved_context)
