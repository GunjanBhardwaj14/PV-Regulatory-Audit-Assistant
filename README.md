# Pharmacovigilance Regulatory Compliance & Audit Assistant

An AI-powered Retrieval-Augmented Generation (RAG) web application that audits draft Individual Case Safety Reports (ICSRs) against **EMA Good Pharmacovigilance Practices (GVP) Module VI (Rev 2)** guidelines, returning compliance verdicts with verifiable PDF page citations.

**Live Demo:** [Launch App](https://pv-regulatory-audit-assistant-nc43zutxuhblt8jv3xmtm6.streamlit.app/)

## Key Features
* **Regulatory RAG Pipeline:** Indexes EMA GVP Module VI (Rev 2) into a persistent ChromaDB vector store (`chunk_size=750`, `chunk_overlap=200`) using Hugging Face `all-MiniLM-L6-v2` embeddings.
* **Grounded Compliance Auditing:** Retrieves the top 5 semantically relevant regulatory clauses and evaluates draft safety reports via Gemini Flash-Lite (`langchain-google-genai`) under strict anti-hallucination prompt guardrails.
* **Audit Traceability:** Supports `.txt` report uploads, custom compliance rule queries, and an expandable **Retrieved EMA GVP Module VI Sources** panel displaying exact PDF page citations.

## Tech Stack
* **Framework & UI:** Python, Streamlit
* **RAG & Embeddings:** LangChain, ChromaDB, `sentence-transformers` (`all-MiniLM-L6-v2`), `pypdf`
* **LLM:** Google Gemini (`gemini-3.5-flash-lite`)

## Project Structure
* `ingest.py` — Parses the EMA GVP Module VI PDF, splits regulatory text into overlapping chunks, and builds `./chroma_db`.
* `retrieve.py` — Standalone script to verify top-5 vector retrieval and PDF page metadata.
* `app.py` — Streamlit web application combining vector retrieval, LLM compliance auditing, and source inspection.
* `pv_data/` — Regulatory reference document (EMA GVP Module VI Rev 2).

## Run Locally
1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/GunjanBhardwaj14/PV-Regulatory-Audit-Assistant.git
   cd PV-Regulatory-Audit-Assistant
   pip install -r requirements.txt
