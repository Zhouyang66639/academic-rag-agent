# Academic RAG Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-1.1-green" />
  <img src="https://img.shields.io/badge/Search-MultiQuery%2BBM25%2BFAISS%2BRRF-blueviolet" />
  <img src="https://img.shields.io/badge/Embedding-FastEmbed_ONNX-brightgreen" />
  <img src="https://img.shields.io/badge/Memory-3--Tier_Persistent-orange" />
  <img src="https://img.shields.io/badge/Free_API-SiliconFlow%20%7C%20Groq-success" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <b>A Research-Grade Memory-Augmented LLM Agent for Academic Literature Analysis</b><br>
  面向学术科研文献的记忆增强 LLM Agent — LangGraph · MultiQuery · Hybrid RAG · Persistent Memory
</p>

---

## What It Does

A production-grade AI agent that combines **retrieval-augmented generation**, **multi-query hybrid search**, **persistent three-tier memory**, and **automated survey writing** into a unified CLI tool for academic research.

```
# Index 100 arXiv papers in seconds (deduplication built-in)
You > bulk_search RAG retrieval augmented generation LLM 100
[✓ 100 papers indexed | duplicates automatically skipped]

# Questions in ANY language — agent queries tools in English automatically
You > RAG系统最新的检索优化方法有哪些？
Agent → [tool call: "recent retrieval optimization methods RAG 2024"]
Agent > 根据最新论文，主要方法包括：(1) MultiQueryRetrieval...
        来源: [CRAG, arXiv:2401.15884] [Self-RAG, arXiv:2310.11511]

# Save research insights to permanent memory
You > remember HyDE: embed hypothetical answer instead of query for better recall
[✓ Saved to memory #1 — injected into system prompt next startup]

# Generate a full survey (grounded in indexed papers)
You > survey Retrieval-Augmented Generation for Large Language Models
[Writing 8 sections via Map-Reduce, each grounded by hybrid retrieval...]
[✓ Saved: survey_Retrieval-Augmented_Generation_20260417_1800.md]
```

---

## System Architecture

![System Architecture](docs/images/architecture.png)

---

## Key Features

| Feature | Implementation | Research Basis |
|---------|---------------|----------------|
| **MultiQuery Retrieval** | LLM generates 3 query variants → merge & dedup | Ma et al. (2023) — +20-40% recall |
| **Hybrid Search** | BM25 (sparse) + FAISS MMR (dense) + RRF fusion | Cormack et al., SIGIR 2009 |
| **LangGraph Agent** | `create_react_agent` + `MemorySaver` | Replaces deprecated `AgentExecutor` |
| **FastEmbed** | `BAAI/bge-small-en-v1.5` via ONNX Runtime | No PyTorch — ~150 MB vs ~1 GB |
| **Chain-of-Thought Prompt** | 5-step reasoning + citation guidelines | Wei et al., NeurIPS 2022 |
| **arXiv Deduplication** | Persistent ID set across sessions | Prevents vector store bloat |
| **Bulk Indexing** | arXiv API → 100+ paper abstracts | Rate-limited, progress bar |
| **Survey Writer** | Map-Reduce pipeline, 8 sections | RAG-grounded, saves to `.md` |
| **3-Tier Memory** | Working / Episodic / Semantic | Tulving (1972); Park et al. (2023) |
| **Free API** | SiliconFlow Qwen / Groq Llama | Zero cost to start |

---

## Retrieval Pipeline

![Hybrid Search Pipeline](docs/images/hybrid_search.png)

### Multi-Query → Hybrid → RRF

The full retrieval stack for every query:

```
User question (any language)
       │
       ▼  [translated to English by Agent]
MultiQueryRetriever  ─── generates 3 query variants via LLM
       │                  e.g. "RAG" → ["retrieval augmented generation",
       │                                "dense retrieval LLM grounding",
       │                                "knowledge-grounded language model"]
       ▼
  For each variant:
  ┌── BM25 keyword search   → ranked list A
  └── FAISS MMR dense search → ranked list B
              │
              ▼
       RRF Fusion  score = Σ 1 / (60 + rank + 1)
              │
              ▼
        Deduplicated Top-K results
```

- **BM25** — exact keyword match, best for technical terms & author names
- **FAISS MMR** — semantic similarity with diversity (prevents redundant results)
- **RRF** — rank fusion without score calibration (`k=60`, Cormack et al. SIGIR 2009)
- **MultiQuery** — 3 phrasings per query, published to improve recall by 20–40%

---

## Three-Tier Memory System

![Memory System](docs/images/memory_system.png)

| Tier | Type | Scope | Commands |
|------|------|-------|---------|
| 1 — Working | LangGraph `MemorySaver` | Current session | automatic |
| 2 — Episodic | Session archive (JSON) | Permanent, last 10 | `clear_memory`, `exit` |
| 3 — Semantic | User fact notes (JSON) | Permanent | `remember` / `forget` |

**Cross-session memory injection**: On every startup, Tier 2 & 3 facts are injected into the agent's system prompt — the agent "remembers" your past research even after restart.

> Inspired by Tulving (1972) *Episodic & Semantic Memory*, Park et al. (2023) *Generative Agents*, Zhong et al. (2024) *MemoryBank*

---

## Survey Generation Pipeline

![Survey Pipeline](docs/images/survey_pipeline.png)

Map-Reduce workflow: for each of the **8 sections**, the MultiQuery+Hybrid retriever finds the most relevant paper chunks; the LLM writes that section grounded in retrieved context; all sections are assembled into a final Markdown document.

**Output sections**: Abstract · Introduction · Background · Taxonomy · Methods · Applications · Challenges & Limitations · Conclusion & Future Directions

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
# Edit .env — fill in OPENAI_API_KEY
```

| Provider | Sign Up | Recommended Free Model |
|----------|---------|------------------------|
| [SiliconFlow](https://cloud.siliconflow.cn) | Phone / Email | `Qwen/Qwen2.5-72B-Instruct` |
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
| `load_dir <dir>` | Index all docs in a folder |

### arXiv Commands
| Command | Description |
|---------|-------------|
| `bulk_search <topic> [N]` | Fetch & index N abstracts from arXiv (default 50, dedup enabled) |

```
bulk_search retrieval augmented generation survey 100
```

### Survey Generation
| Command | Description |
|---------|-------------|
| `survey <topic>` | Generate a complete 8-section survey paper (Markdown) |
| `survey <topic> --out file.md` | Save to a specific file |

```
bulk_search RAG large language model 80
survey Retrieval-Augmented Generation for LLMs
```

### Memory Commands
| Command | Description |
|---------|-------------|
| `remember <note>` | Save a fact to permanent memory (survives restart) |
| `memories` | Show all saved facts & session archive |
| `forget <id>` | Delete a memory by ID |
| `clear_memory` | Auto-archive session summary and reset chat |

### System Commands
| Command | Description |
|---------|-------------|
| `status` | Show KB size, memory stats, paths |
| `clear_db` | Wipe the vector store |
| `help` | Show all commands |
| `exit` | Auto-save session summary and quit |

---

## Project Structure

```
academic-rag-agent/
├── main.py                       # CLI entry point — all commands
├── requirements.txt
├── environment.yml               # Conda environment
├── .env.example                  # Config template (no secrets)
├── docs/
│   └── images/                   # Architecture diagrams
└── src/
    ├── agent/
    │   └── agent.py              # LangGraph ReAct agent + CoT prompt
    ├── rag/
    │   ├── document_loader.py    # PDF/TXT/MD loader with file dedup
    │   ├── vector_store.py       # FAISS + MultiQuery + FastEmbed + arXiv dedup
    │   └── hybrid_retriever.py   # BM25 + FAISS + RRF fusion
    ├── memory/
    │   └── persistent_memory.py  # Three-tier persistent memory
    └── tools/
        ├── arxiv_tool.py         # arXiv live search (LangChain tool)
        ├── bulk_arxiv.py         # Bulk metadata fetcher (100+ papers)
        └── survey_writer.py      # Map-Reduce survey generation
```

---

## Tech References

| Technology | Reference |
|-----------|-----------|
| RAG | Lewis et al., *RAG for Knowledge-Intensive NLP* (NeurIPS 2020) |
| ReAct Agent | Yao et al., *ReAct: Synergizing Reasoning and Acting* (ICLR 2023) |
| MultiQuery Retrieval | Ma et al., *Query Rewriting for RAG* (2023) |
| Chain-of-Thought | Wei et al., *Chain-of-Thought Prompting* (NeurIPS 2022) |
| RRF Fusion | Cormack, Clarke & Buettcher, *RRF outperforms Condorcet* (SIGIR 2009) |
| MMR Retrieval | Carbonell & Goldstein, *The Use of MMR* (SIGIR 1998) |
| Memory System | Tulving (1972); Park et al., *Generative Agents* (2023) |
| LangGraph | [LangGraph Docs](https://langchain-ai.github.io/langgraph/) |

---

## Roadmap

- [x] Hybrid BM25 + FAISS + RRF retrieval
- [x] MultiQueryRetriever (Ma et al. 2023, +20-40% recall)
- [x] arXiv paper deduplication across sessions
- [x] LangGraph `create_react_agent` with Chain-of-Thought prompt
- [x] FastEmbed (ONNX, no PyTorch)
- [x] Bulk arXiv indexing (100+ papers)
- [x] Map-Reduce survey generation (8 sections)
- [x] Three-tier persistent memory (Working / Episodic / Semantic)
- [ ] Cross-encoder reranker (BGE-reranker, Nogueira et al. 2019)
- [ ] HyDE — Hypothetical Document Embeddings (Gao et al. 2022)
- [ ] RAGAS automated evaluation (faithfulness / relevancy metrics)
- [ ] Streamlit / Gradio web UI
- [ ] Multi-agent collaboration (researcher + critic + writer)

---

## License

MIT License
