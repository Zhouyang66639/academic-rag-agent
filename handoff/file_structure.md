# 文件结构说明

## 项目根目录：`academic-rag-agent/`

```
academic-rag-agent/
│
├── .env                    ← 【关键】API Key / 模型配置 (不入 Git)
├── .env.example            ← 配置模板
├── main.py                 ← 入口：Rich CLI 交互循环
├── requirements.txt        ← 依赖列表
├── check_models.py         ← 工具：测试 API 模型可用性
├── diagnose.py             ← 工具：调试用(可删)
│
├── handoff/                ← 【本文档目录】AI 交接文档
│   ├── README.md           ← 主交接文档（先读这个！）
│   ├── quick_test.py       ← 快速验证脚本
│   └── file_structure.md  ← 本文件
│
├── src/
│   ├── agent/
│   │   └── agent.py       ← 【核心】LangGraph ReAct Agent
│   │
│   ├── rag/
│   │   ├── vector_store.py ← FAISS+BM25 混合检索
│   │   ├── document_loader.py ← PDF/TXT 加载
│   │   └── embeddings.py  ← FastEmbed 本地嵌入
│   │
│   ├── tools/
│   │   └── arxiv_tool.py  ← arXiv API 工具（已加 _clean()）
│   │
│   └── memory/
│       └── persistent_memory.py ← 跨会话记忆（JSON 文件）
│
├── vector_store/           ← FAISS 索引文件（运行时生成）
└── memory/                 ← 持久记忆 JSON 文件（运行时生成）
```

## 最重要的三个文件

1. **`.env`** — API 配置，每次换提供商都改这里
2. **`src/agent/agent.py`** — 所有 Agent 逻辑，bug 在 `_build_graph()` 和 `chat()`
3. **`src/tools/arxiv_tool.py`** — ArXiv 工具，已加 surrogate 清洗

## 已确认工作正常的组件

- ✅ FAISS 向量存储（FastEmbed BAAI/bge-small-en-v1.5）
- ✅ BM25 混合检索
- ✅ PDF/TXT 文件加载
- ✅ arXiv 实时搜索
- ✅ PersistentMemory 跨会话记忆
- ✅ Rich CLI 界面
- ✅ Windows UTF-8 编码修复

## 尚未工作的组件

- ❌ `agent.chat()` — LangGraph 内部引入 surrogate，每次返回 Error
