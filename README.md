# 📄🔎 Multimodal RAG

A step-by-step **multimodal Retrieval-Augmented Generation (RAG)** pipeline built in Python.  
It ingests research papers (PDFs), summarizes both **text** and **figures**, indexes them with **Chroma**, and enables **multimodal question-answering** using `gpt-4o-mini`.  
Includes a **Streamlit UI** to visualize answers, retrieved text snippets, and figure thumbnails.

---

## ✨ Features

- **PDF ingestion** → extracts page text + embedded images
- **Summarization** → text pages (concise, fact-dense) and image captions (figure insights)
- **Indexing** → Multi-Vector Retriever (summaries → parent docs)
- **QA** → answers grounded in text & images, powered by `gpt-4o-mini`
- **Frontend** → Streamlit app for interactive exploration
