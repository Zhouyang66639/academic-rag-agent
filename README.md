# Academic RAG Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-1.1-green" />
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-orange" />
  <img src="https://img.shields.io/badge/Hybrid_Search-BM25%2BFAISS%2BRRF-blueviolet" />
  <img src="https://img.shields.io/badge/Embedding-FastEmbed_ONNX-brightgreen" />
  <img src="https://img.shields.io/badge/Free_API-SiliconFlow%20%7C%20Groq-success" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <b>A Memory-Augmented LLM Agent for Academic Literature Analysis</b><br>
  面向学术科研文献的记忆增强 LLM Agent · v2.0
</p>

---

## What It Does

Upload your research papers, ask questions in natural language, and get answers grounded in your documents — with conversation memory and real-time arXiv search.

```
You > load papers/attention_is_all_you_need.pdf
[47 chunks indexed — Hybrid BM25+FAISS retriever activated]

You > What is the role of Multi-Head Attention?

Agent > Based on [attention_is_all_you_need.pdf], Multi-Head Attention allows
        the model to jointly attend to information from different representation
        subspaces at different positions...

You > How does this compare to earlier attention mechanisms?

Agent > (remembers context from previous turn via LangGraph MemorySaver)
        Compared to additive attention used in Bahdanau et al...

You > Search for recent papers on RAG

Agent > Found 5 papers on arXiv:
        [1] Engineering the RAG Stack — arXiv:2601.05264
        ...
```

---

## Architecture

```mermaid
flowchart TD
    A([User Input]) --> B

    subgraph LangGraph_StateGraph ["LangGraph StateGraph (ReAct Loop)"]
        B[Agent Node\nLLM Reasoning] -->|tool call| C{Tool Router}
        C -->|local docs| D[search_uploaded_papers]
        C -->|latest papers| E[arxiv_search_tool]
        C -->|general Q&A| F[Direct Response]
        D --> G[Observation]
        E --> G
        G --> B
    end

    subgraph Hybrid_Retrieval ["Hybrid Retrieval Pipeline"]
        D --> H[BM25\nSparse Retrieval]
        D --> I[FAISS MMR\nDense Retrieval]
        H --> J[RRF Fusion\n1÷60+rank]
        I --> J
        J --> K[(Top-K Docs)]
    end

    subgraph Memory ["MemorySaver Checkpointing"]
        L[(Thread State\nMessage History)] -->|thread_id| B
        B -->|checkpoint| L
    end

    F --> M([Final Answer])
    K --> M

    style A fill:#4f46e5,color:#fff
    style M fill:#059669,color:#fff
    style J fill:#d97706,color:#fff
    style L fill:#7c3aed,color:#fff
    style B fill:#1e40af,color:#fff
```

---

## Key Features

| Feature | Implementation | Notes |
|---------|---------------|-------|
| **RAG** | PDF/TXT/MD → chunked → FAISS index | RecursiveCharacterTextSplitter |
| **Hybrid Search** | BM25 (sparse) + FAISS MMR (dense) + RRF | Cormack et al., SIGIR 2009 |
| **Memory** | LangGraph MemorySaver + thread\_id | Per-session checkpointing |
| **Agent** | LangGraph `create_react_agent` (v2) | Replaces deprecated AgentExecutor |
| **Embedding** | FastEmbed `BAAI/bge-small-en-v1.5` | ONNX Runtime, **no PyTorch** |
| **Free-Friendly** | SiliconFlow (Qwen) / Groq (Llama) | Zero API cost to get started |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/academic-rag-agent.git
cd academic-rag-agent

# Create isolated conda environment
conda env create -f environment.yml
conda activate rag-agent
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and fill in your API key
```

**Recommended free options:**

| Provider | Sign Up | Free Model |
|----------|---------|-----------|
| [SiliconFlow](https://cloud.siliconflow.cn) | Phone/Email | `Qwen/Qwen2.5-7B-Instruct` |
| [Groq](https://console.groq.com) | Google account | `llama-3.3-70b-versatile` |

Example `.env` for SiliconFlow:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Run

```bash
# Windows PowerShell
conda activate rag-agent
cd path/to/academic-rag-agent
python main.py
```

---

## Usage

| Command | Description |
|---------|-------------|
| `load <path/to/paper.pdf>` | Index a PDF into the knowledge base |
| `load_dir <path/to/folder>` | Index all documents in a folder |
| `status` | Show knowledge base size and memory usage |
| `clear_memory` | Reset conversation history |
| `clear_db` | Wipe the vector store |
| `help` | Show all commands |
| `exit` | Quit |

---

## RAG Pipeline Detail

```mermaid
flowchart LR
    subgraph Indexing
        A[PDF / TXT / MD] --> B[RecursiveCharacterTextSplitter\nchunk=1000, overlap=200]
        B --> C[HuggingFace Embeddings\nall-MiniLM-L6-v2]
        C --> D[(FAISS Index\nLocal Persistence)]
    end

    subgraph Retrieval
        E([Query]) --> F[MMR Search\nk=4, fetch_k=12]
        D --> F
        F --> G[Top-K Chunks]
    end

    style D fill:#d97706,color:#fff
```

---

## Project Structure

```
academic-rag-agent/
├── main.py                    # CLI entry point
├── requirements.txt
├── environment.yml            # Conda environment
├── .env.example               # Config template
├── STARTUP.md                 # Personal startup guide
└── src/
    ├── rag/
    │   ├── document_loader.py # Load & chunk documents
    │   └── vector_store.py    # FAISS management
    ├── agent/
    │   └── agent.py           # ReAct agent core
    └── tools/
        └── arxiv_tool.py      # arXiv search tool
```

---

## Tech References

- **RAG**: Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020)
- **ReAct**: Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (ICLR 2023)
- **MMR**: Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking for Reordering Documents* (SIGIR 1998)

---

## Roadmap

- [ ] Gradio / Streamlit web UI
- [ ] Multi-modal support (figures, equations)
- [ ] Zotero library integration
- [ ] Literature review draft generation
- [ ] Multi-agent collaboration

---

## License

MIT License
