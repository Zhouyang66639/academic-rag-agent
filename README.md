# Academic RAG Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-1.1-green" />
  <img src="https://img.shields.io/badge/Hybrid_Search-BM25%2BFAISS%2BRRF-blueviolet" />
  <img src="https://img.shields.io/badge/Embedding-FastEmbed_ONNX-brightgreen" />
  <img src="https://img.shields.io/badge/Memory-3--Tier_Persistent-orange" />
  <img src="https://img.shields.io/badge/Free_API-SiliconFlow%20%7C%20Groq-success" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <b>A Memory-Augmented LLM Agent for Academic Literature Analysis</b><br>
  面向学术科研文献的记忆增强 LLM Agent — powered by LangGraph &amp; Hybrid RAG
</p>

---

## What It Does

A production-grade AI agent that combines **retrieval-augmented generation**, **hybrid search**, **persistent memory**, and **automated survey writing** into a unified CLI tool for academic research.

```
# Index 100 arXiv papers in seconds
You > bulk_search RAG retrieval augmented generation LLM 100
[100 papers indexed via BM25 + FAISS hybrid retriever]

# Ask questions grounded in real papers
You > What are the main limitations of current RAG systems?
Agent > Based on indexed papers, the key limitations are: (1) retrieval quality...

# Save important insights to persistent memory
You > remember RRF fusion formula: score = sum(1 / (k + rank_i))
[Saved to memory #1]

# Generate a full survey paper automatically
You > survey Retrieval-Augmented Generation for Large Language Models
[Writing 8 sections via Map-Reduce pipeline...]
[Saved: survey_Retrieval-Augmented_Generation_20260417_1800.md]

# Memory persists across restarts
[Next session: fact notes auto-injected into system prompt]
```

---

## System Architecture

```mermaid
flowchart TD
    A([User Input]) --> CMD{Command\nRouter}

    CMD -->|question| B
    CMD -->|bulk_search| J
    CMD -->|survey| K
    CMD -->|remember| L

    subgraph Agent ["LangGraph ReAct Agent"]
        B[Agent Node\nLLM Reasoning] -->|tool call| C{Tool\nRouter}
        C -->|local docs| D[search_uploaded_papers]
        C -->|latest papers| E[arxiv_search_tool]
        C -->|general| F[Direct Answer]
        D --> G[Observation]
        E --> G
        G --> B
    end

    subgraph HybridSearch ["Hybrid Retrieval Pipeline"]
        D --> H1[BM25 Sparse]
        D --> H2[FAISS MMR Dense]
        H1 --> H3["RRF Fusion\n1÷(60+rank)"]
        H2 --> H3
        H3 --> H4[(Top-K Docs)]
    end

    subgraph Memory ["Three-Tier Memory System"]
        M1[("Tier 1\nWorking Memory\nLangGraph MemorySaver")]
        M2[("Tier 2\nEpisodic Memory\nSession Archive JSON")]
        M3[("Tier 3\nSemantic Memory\nUser Fact Notes JSON")]
        M1 -.->|"auto-checkpoint\n(session)"| M2
        M3 -.->|"inject into\nsystem prompt"| B
    end

    subgraph BulkIndex ["Bulk arXiv Indexer"]
        J --> J1[arXiv API\nfetch N abstracts]
        J1 --> J2[Document Objects]
        J2 --> H1
        J2 --> H2
    end

    subgraph SurveyPipeline ["Survey Generation (Map-Reduce)"]
        K --> K1[Section 1: Abstract]
        K --> K2[Section 2: Intro]
        K --> K3[Section 3-7: ...]
        K1 & K2 & K3 --> K4[Markdown Assembly]
        K4 --> K5([📄 survey_.md])
        H4 -.->|context chunks| K1
        H4 -.->|context chunks| K2
        H4 -.->|context chunks| K3
    end

    L --> M3
    F --> M([Final Answer])
    H4 --> M

    style A fill:#4f46e5,color:#fff
    style M fill:#059669,color:#fff
    style H3 fill:#d97706,color:#fff
    style M1 fill:#7c3aed,color:#fff
    style M2 fill:#6d28d9,color:#fff
    style M3 fill:#5b21b6,color:#fff
    style K5 fill:#0891b2,color:#fff
```

---

## Key Features

| Feature | Implementation | Technical Detail |
|---------|---------------|-----------------|
| **Hybrid Search** | BM25 (sparse) + FAISS MMR (dense) + RRF | Cormack et al., SIGIR 2009 |
| **LangGraph Agent** | `create_react_agent` + `MemorySaver` | Replaces deprecated `AgentExecutor` |
| **FastEmbed** | `BAAI/bge-small-en-v1.5` ONNX Runtime | No PyTorch — ~150MB vs ~1GB |
| **Bulk Indexing** | arXiv API → 100+ paper abstracts | Respects rate limits, progress bar |
| **Survey Writer** | Map-Reduce pipeline, 8 sections | RAG-grounded, saves to `.md` |
| **3-Tier Memory** | Working / Episodic / Semantic | Persists across restarts as JSON |
| **Free API** | SiliconFlow Qwen / Groq Llama | Zero cost to start |

---

## Quick Start

### 1. Clone & Create Environment

```bash
git clone https://github.com/YOUR_USERNAME/academic-rag-agent.git
cd academic-rag-agent

conda env create -f environment.yml
conda activate rag-agent
```

### 2. Configure API Key

```bash
copy .env.example .env
# Edit .env and fill in your API key
```

| Provider | Sign Up | Free Model |
|----------|---------|------------|
| [SiliconFlow](https://cloud.siliconflow.cn) | Phone/Email | `Qwen/Qwen2.5-72B-Instruct` |
| [Groq](https://console.groq.com) | Google account | `llama-3.3-70b-versatile` |

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen2.5-72B-Instruct
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### 3. Run

```powershell
conda activate rag-agent
cd path\to\academic-rag-agent
python main.py
```

---

## Command Reference

### Document Commands
| Command | Description |
|---------|-------------|
| `load <path>` | Index a PDF / TXT / MD file |
| `load_dir <dir>` | Index all documents in a folder |

### arXiv Commands
| Command | Description |
|---------|-------------|
| `bulk_search <topic> [N]` | Fetch & index N abstracts from arXiv (default 50) |

Example:
```
bulk_search retrieval augmented generation survey 100
```

### Survey Generation
| Command | Description |
|---------|-------------|
| `survey <topic>` | Generate a complete survey paper (Markdown) |
| `survey <topic> --out file.md` | Save to a specific file |

Example workflow:
```
bulk_search RAG large language model 80
survey Retrieval-Augmented Generation for LLMs
```

### Memory Commands
| Command | Description |
|---------|-------------|
| `remember <note>` | Save a fact/insight to permanent memory |
| `memories` | Show all saved facts & session archive |
| `forget <id>` | Delete a memory note by ID |
| `clear_memory` | Archive current session and reset chat |

### System Commands
| Command | Description |
|---------|-------------|
| `status` | Show knowledge base & memory stats |
| `clear_db` | Wipe the vector store |
| `help` | Show all commands |
| `exit` | Auto-save session summary and quit |

---

## Memory System Architecture

```mermaid
flowchart LR
    subgraph Tier1 ["Tier 1 — Working Memory"]
        A[LangGraph MemorySaver\nIn-session message history\nAuto-managed by StateGraph]
    end

    subgraph Tier2 ["Tier 2 — Episodic Memory"]
        B[Session Archive\nmemory/memory.json\nAuto-saved on exit & clear]
    end

    subgraph Tier3 ["Tier 3 — Semantic Memory"]
        C[User Fact Notes\nmemory/memory.json\nremember command]
    end

    A -->|session ends| B
    B & C -->|next startup| D[System Prompt\nInjection]
    D --> E([Agent has context\nfrom past sessions])

    style A fill:#7c3aed,color:#fff
    style B fill:#6d28d9,color:#fff
    style C fill:#5b21b6,color:#fff
    style E fill:#059669,color:#fff
```

**Reference:** Inspired by Tulving (1972) *Episodic & Semantic Memory*, Park et al. (2023) *Generative Agents*, and Zhong et al. (2024) *MemoryBank*.

---

## Survey Generation Pipeline

```mermaid
sequenceDiagram
    participant User
    participant SurveyWriter
    participant HybridRetriever
    participant LLM

    User->>SurveyWriter: survey <topic>
    Note over SurveyWriter: 8 sections to write

    loop For each section (Abstract → Conclusion)
        SurveyWriter->>HybridRetriever: "{topic} {section_name}"
        HybridRetriever-->>SurveyWriter: top-5 BM25+FAISS+RRF chunks
        SurveyWriter->>LLM: context + section writing prompt
        LLM-->>SurveyWriter: section draft
    end

    SurveyWriter->>User: assembled survey.md (~3500 words)
```

---

## RAG Pipeline Detail

```mermaid
flowchart LR
    subgraph Indexing
        A[PDF / TXT / MD\narXiv Abstract] --> B["RecursiveCharacterTextSplitter\nchunk=1000, overlap=200"]
        B --> C["FastEmbed\nBAAI/bge-small-en-v1.5\n(ONNX Runtime)"]
        C --> D[(FAISS Index\nLocal Persistence)]
        B --> E[(Raw Docs\ndocs.pkl\nfor BM25)]
    end

    subgraph Retrieval
        F([Query]) --> G[BM25 Search\nsparse / keyword]
        F --> H[FAISS MMR\ndense / semantic]
        D --> H
        E --> G
        G & H --> I["RRF Fusion\nscore = Σ 1/(60+rank)"]
        I --> J[Top-K Results]
    end

    style D fill:#d97706,color:#fff
    style E fill:#92400e,color:#fff
    style I fill:#1e40af,color:#fff
```

---

## Project Structure

```
academic-rag-agent/
├── main.py                       # CLI entry point (all commands)
├── requirements.txt
├── environment.yml               # Conda environment
├── .env.example                  # Config template (no secrets)
└── src/
    ├── agent/
    │   └── agent.py              # LangGraph ReAct agent
    ├── rag/
    │   ├── document_loader.py    # PDF/TXT/MD loader with dedup
    │   ├── vector_store.py       # FAISS + FastEmbed + persistence
    │   └── hybrid_retriever.py   # BM25 + FAISS + RRF fusion
    ├── memory/
    │   └── persistent_memory.py  # Three-tier memory system
    └── tools/
        ├── arxiv_tool.py         # arXiv search (LangChain tool)
        ├── bulk_arxiv.py         # Bulk abstract fetcher (100+)
        └── survey_writer.py      # Map-Reduce survey generation
```

---

## Tech References

| Technology | Reference |
|-----------|-----------|
| RAG | Lewis et al., *RAG for Knowledge-Intensive NLP* (NeurIPS 2020) |
| ReAct Agent | Yao et al., *ReAct: Synergizing Reasoning and Acting* (ICLR 2023) |
| RRF Fusion | Cormack, Clarke & Buettcher, *RRF outperforms Condorcet* (SIGIR 2009) |
| MMR Retrieval | Carbonell & Goldstein, *The Use of MMR* (SIGIR 1998) |
| Memory System | Tulving (1972); Park et al., *Generative Agents* (2023) |
| LangGraph | [LangGraph Docs](https://langchain-ai.github.io/langgraph/) |

---

## Roadmap

- [x] Hybrid BM25 + FAISS + RRF retrieval
- [x] LangGraph `create_react_agent`
- [x] FastEmbed (ONNX, no PyTorch)
- [x] Bulk arXiv indexing (100+ papers)
- [x] Map-Reduce survey generation
- [x] Three-tier persistent memory
- [ ] Streamlit / Gradio web UI
- [ ] Zotero library integration
- [ ] Multi-agent collaboration (researcher + critic + writer)
- [ ] PDF figure and equation extraction

---

## License

MIT License
