# Corrective RAG using LangGraph

A production-style **Corrective Retrieval-Augmented Generation (CRAG)** application built with **LangGraph**, **LangChain**, **ChromaDB**, **Groq LLM**, **HuggingFace Embeddings**, **Tavily Search**, and **Streamlit**.

Unlike a traditional RAG pipeline, this project evaluates the quality of retrieved documents before generating a response. Depending on the retrieval quality, it intelligently decides whether to:

- Answer directly from retrieved documents
- Combine retrieved documents with web search
- Ignore retrieved documents and rely completely on web search

This significantly improves answer quality and reduces hallucinations.

---

# Architecture

![Corrective RAG Architecture](Architecture.png)

---

# Retrieval Decisions

| Decision | Description |
|-----------|-------------|
| **Correct** | Retrieved documents are sufficient to answer the query. |
| **Ambiguous** | Retrieved documents are partially relevant. Retrieve additional information using Tavily Search and merge both contexts. |
| **Incorrect** | Retrieved documents are irrelevant. Ignore them and generate the answer completely from web search. |

---

# Project Structure

```text
Corrective_RAG/
│
├── app.py                          # Streamlit application
├── Corrective_RAG_Architecture.py  # LangGraph workflow
├── Rag_flow.py                     # Chroma retrieval
├── llm.py                          # LLM configuration
├── state.py                        # Graph state
│
├── nodes/
│   ├── get_relevant_docs.py        # Retrieve vector documents
│   ├── eval_doc_node.py            # Evaluate retrieval quality
│   ├── ambiguous_retrieval.py      # Handle ambiguous retrieval
│   ├── web_search.py               # Tavily web search
│   ├── refine_context.py           # Merge retrieved and web documents
│   └── generate_answer.py          # Final answer generation
│
├── uploaded_docs/                  # User uploaded knowledge base
├── chroma_db/                      # Chroma vector database
│
├── basic_RAG/                      # Basic RAG implementation
├── env_template.txt
├── requirements.txt
└── README.md
```

---

# LangGraph Nodes

## 1. Retrieve Documents

**nodes/get_relevant_docs.py**

- Searches ChromaDB
- Retrieves top matching document chunks
- Stores retrieved documents in graph state

---

## 2. Evaluate Retrieval

**nodes/eval_doc_node.py**

Uses an LLM to classify retrieval into one of three categories:

- Correct
- Ambiguous
- Incorrect

---

## 3. Ambiguous Retrieval

**nodes/ambiguous_retrieval.py**

When retrieved documents are only partially useful:

- Extracts useful retrieved documents
- Performs Tavily web search
- Stores both retrieved and web documents separately

---

## 4. Web Search

**nodes/web_search.py**

When retrieval is classified as incorrect:

- Ignores retrieved documents
- Retrieves fresh information from Tavily Search
- Uses web results for answer generation

---

## 5. Refine Context

**nodes/refine_context.py**

Creates the final context for the LLM by combining:

- Relevant retrieved documents
- Web search documents

depending on the retrieval decision.

---

## 6. Generate Answer

**nodes/generate_answer.py**

Generates the final response using the refined context.

---

# Features

- Corrective RAG (CRAG)
- LangGraph state machine
- Dynamic retrieval evaluation
- Intelligent web search fallback
- Context refinement
- ChromaDB vector store
- HuggingFace embeddings
- Tavily Search integration
- Streamlit user interface
- User document upload support
- Modular node-based architecture
- Reduced hallucinations

---

# Tech Stack

- Python
- LangGraph
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq (Llama 3.3 70B)
- Tavily Search API
- Streamlit

---

# Installation

Clone the repository

```bash
git clone https://github.com/ChanduAnnavarapu/Corrective_RAG.git
```

Move into the project directory

```bash
cd Corrective_RAG
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.\.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file using `env_template.txt`.

```env
API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

# Run the Application

Launch the Streamlit application

```bash
streamlit run app.py
```

---

# Build the Vector Database

Upload your documents through the Streamlit interface.

The application automatically:

- Splits documents into chunks
- Generates embeddings
- Stores them in ChromaDB
- Uses them for future retrieval

---