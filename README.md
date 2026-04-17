# 🎓 Academic RAG Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-0.3-green?logo=langchain" />
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-orange" />
  <img src="https://img.shields.io/badge/LLM-DeepSeek%20%7C%20OpenAI-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <b>面向学术科研文献的记忆增强 LLM Agent</b><br>
  A Memory-Augmented LLM Agent for Academic Literature Analysis
</p>

---

## ✨ 项目概述

本项目实现了一个专为**学术科研场景**设计的对话式 AI Agent，能够：

- 📚 **读懂论文**：加载 PDF / TXT / Markdown 格式文献，自动分块向量化
- 🔍 **语义检索**：基于 FAISS 向量数据库实现精准 RAG 检索
- 🧠 **记住上下文**：滑动窗口记忆机制，支持多轮追问
- 🌐 **实时搜索**：调用 arXiv API 搜索最新相关论文
- 🛠️ **工具调用**：LLM 自主决策何时检索本地文献、何时搜索网络

**应用场景**：文献综述辅助、论文精读提问、跨文献对比分析、研究思路探索

---

## 🏗️ 系统架构

```
用户输入 (Query)
     │
     ▼
┌─────────────────────────────────────┐
│          Tool-Calling Agent          │
│  (LangChain AgentExecutor + LLM)    │
│                                     │
│   ┌──────────┐   ┌───────────────┐  │
│   │  Tool 1  │   │    Tool 2     │  │
│   │ 本地文档  │   │  arXiv 搜索   │  │
│   │  RAG检索  │   │  实时论文搜索  │  │
│   └────┬─────┘   └──────┬────────┘  │
│        │                │           │
│        ▼                ▼           │
│   ┌─────────┐      ┌─────────┐      │
│   │  FAISS  │      │ arXiv   │      │
│   │ 向量数据库│      │   API   │      │
│   └─────────┘      └─────────┘      │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────┐
│  Conversation   │  ← 滑动窗口记忆 (Window Memory)
│     Memory      │     保留最近 N 轮对话上下文
└─────────────────┘
     │
     ▼
   LLM 回答生成 (DeepSeek / GPT-4o-mini)
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/academic-rag-agent.git
cd academic-rag-agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

> 💡 **推荐使用 DeepSeek API**（与 OpenAI 格式完全兼容，价格约为 GPT-4o 的 1/30）
> 申请地址：https://platform.deepseek.com/

### 4. 运行

```bash
python main.py
```

---

## 💬 使用示例

```
🎓 Academic RAG Agent  v1.0

你 > load papers/attention_is_all_you_need.pdf
📄 加载文件: attention_is_all_you_need.pdf (PDF 论文)
✅ 完成: 共生成 47 个文本块
🔄 正在向量化 47 个文本块...
✅ 向量化完成: 数据库现共有 47 个向量块

你 > Transformer 中 Multi-Head Attention 的作用是什么？

🤖 Agent (历史: 0 轮)
根据 [attention_is_all_you_need.pdf]，Multi-Head Attention 的核心作用是...
允许模型在不同的表示子空间中并行关注来自不同位置的信息。具体而言：
1. **多头设计**：将 Q/K/V 投影到 h 个不同的低维空间分别计算注意力...
2. **信息整合**：将 h 个注意力结果拼接后再次投影...

你 > 搜索一下最新的 RAG 综述论文

🤖 Agent (历史: 1 轮)
找到 5 篇相关论文：
[1] A Survey on Retrieval-Augmented Generation for Large Language Models...
```

---

## 📂 项目结构

```
academic-rag-agent/
├── main.py                    # 交互式 CLI 入口
├── requirements.txt
├── .env.example               # 配置模板
├── src/
│   ├── rag/
│   │   ├── document_loader.py # 文档加载 & 分块
│   │   └── vector_store.py    # FAISS 向量数据库管理
│   ├── agent/
│   │   └── agent.py           # 核心 Agent（记忆 + 工具调用）
│   └── tools/
│       └── arxiv_tool.py      # arXiv 搜索工具
└── vector_store/              # 本地向量库（自动生成）
```

---

## ⚙️ 核心技术细节

### RAG Pipeline

| 步骤 | 技术选型 | 说明 |
|------|---------|------|
| 文档分块 | `RecursiveCharacterTextSplitter` | chunk_size=1000, overlap=200 |
| 向量化 | `text-embedding-3-small` | OpenAI 嵌入，尺寸 1536 |
| 存储 | FAISS (IVFFlat) | 本地持久化，无需服务器 |
| 检索 | MMR (Maximal Marginal Relevance) | 相关性 + 多样性平衡 |

### Memory 设计

使用 `ConversationBufferWindowMemory`（滑动窗口记忆）：
- 保留最近 **K 轮**对话（默认 K=10）
- 超出窗口的历史自动丢弃，控制 token 消耗
- 支持 `clear_memory` 命令手动重置

### Agent 决策逻辑

LLM 根据问题语义**自主选择工具**：
- 问题涉及已上传文献 → 触发 `search_uploaded_papers`（本地 RAG）
- 问题需要最新论文 → 触发 `arxiv_search_tool`（网络搜索）
- 一般知识问题 → 直接由 LLM 回答

---

## 🗺️ 未来计划

- [ ] 支持多模态（图表、公式识别）
- [ ] 添加 Web UI（基于 Gradio 或 Streamlit）
- [ ] 接入 Zotero 文献管理库
- [ ] 支持生成文献综述草稿
- [ ] 多 Agent 协作（写作 Agent + 检索 Agent）

---

## 📖 相关工作

本项目的设计灵感来源于以下工作：

- **RAG**: Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020)
- **ReAct**: Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (ICLR 2023)
- **MemGPT**: Packer et al., *MemGPT: Towards LLMs as Operating Systems* (2023)

---

## 📄 License

MIT License © 2025 Zhou Yang (NTU)
