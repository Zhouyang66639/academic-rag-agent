# Academic RAG Agent — 功能说明书

> 写给用户和下一个 AI 看的功能全景图

---

## 🎯 这个系统是什么？

**Academic RAG Agent** 是一个面向 AI 学术研究的智能助手，能够：
- 读取你上传的 PDF/论文，建立本地知识库
- 实时搜索 arXiv 最新论文
- 用自然语言（支持中文）进行问答
- 自动生成学术综述
- 跨会话记住你的研究偏好

---

## 📋 完整功能列表

### 一、文档管理功能

| 命令 | 说明 |
|------|------|
| `load <文件路径>` | 加载单个文件（支持 PDF / TXT / MD）到知识库 |
| `load_dir <目录路径>` | 批量加载一个文件夹内所有文档 |
| `status` | 查看知识库中的文档数量、向量数量、内存状态 |
| `clear_db` | 清空向量数据库（需输入 YES 确认） |

**示例：**
```
你: load D:\papers\attention_is_all_you_need.pdf
你: load_dir D:\papers\rag-survey
你: status
```

---

### 二、arXiv 实时搜索功能

| 命令 | 说明 |
|------|------|
| `bulk_search <关键词> [数量]` | 从 arXiv 批量获取论文摘要并索引到知识库，默认 50 篇 |

**示例：**
```
你: bulk_search RAG retrieval augmented generation 100
你: bulk_search LLM agent planning 50
你: bulk_search diffusion model image generation 30
```

执行后：arXiv 摘要自动进入向量知识库，后续问答可引用它们。

---

### 三、智能问答功能

直接输入问题即可（支持中文）：

```
你: What is the core contribution of the Attention paper?
你: RAG 和 fine-tuning 有什么区别？
你: 最近有什么关于 LLM Agent 的进展？
你: 解释一下 RLHF 的原理
```

**Agent 工作流程（内部）：**
1. 判断是否需要搜索（打招呼类不搜索）
2. 先搜索本地知识库（已上传的 PDF + bulk_search 的摘要）
3. 若本地不够，自动搜索 arXiv 实时补充
4. 综合多篇论文给出有引用的回答

---

### 四、学术综述生成功能

| 命令 | 说明 |
|------|------|
| `survey <主题>` | 根据知识库内容生成结构化综述（Markdown 格式） |
| `survey <主题> --out <文件名>` | 生成并保存到指定文件 |

**示例：**
```
你: survey retrieval augmented generation
你: survey LLM agents --out rag_survey_2025.md
你: survey diffusion models for text generation --out diffusion_nlp.md
```

**生成的综述包含：**
- 研究背景与动机
- 主要方法分类
- 代表性论文对比
- 当前挑战与未来方向
- 参考文献列表（带 arXiv 链接）

---

### 五、记忆功能

#### 5.1 本次会话记忆（自动）
- LangGraph `MemorySaver` 自动保存每轮对话
- 同一次运行中可以引用之前说的内容

#### 5.2 永久记忆（手动保存）

| 命令 | 说明 |
|------|------|
| `remember <内容>` | 把一条笔记永久保存，下次启动仍记得 |
| `memories` | 查看所有保存的笔记和历史会话摘要 |
| `forget <id>` | 删除某条笔记（ID 从 memories 中查看） |
| `clear_memory` | 结束当前会话并归档摘要，开启新会话 |

**示例：**
```
你: remember 我的研究方向是多模态 RAG，重点关注 2024 年后的论文
你: remember 导师要求每周汇报一篇顶会论文
你: memories
你: forget 2
```

---

### 六、系统命令

| 命令 | 说明 |
|------|------|
| `status` | 显示知识库向量数、会话轮次、记忆条数 |
| `help` | 显示帮助菜单 |
| `exit` / `quit` / `q` | 自动保存会话摘要并退出 |

---

## 🔍 检索技术细节

本系统使用**混合检索（Hybrid Search）**：

```
用户问题
   ↓
BM25 关键词检索（稀疏，精确匹配）
   +
FAISS 向量检索（密集，语义相似）
   ↓
RRF 融合排序（Reciprocal Rank Fusion）
   ↓
Top-K 最相关结果
```

- **嵌入模型：** `BAAI/bge-small-en-v1.5`（本地运行，无需 API，ONNX 加速）
- **检索数量：** 默认 Top-4（可在 `.env` 调整 `RETRIEVAL_TOP_K`）
- **MultiQueryRetriever：** 当知识库有文档时，自动对每个查询生成 3 个变体，提升召回率

---

## 📂 `.env` 关键配置项

```env
# LLM API（必填）
OPENAI_API_KEY=你的Key
OPENAI_BASE_URL=https://api.iamhc.cn/v1   # 或其他兼容 OpenAI 格式的服务
OPENAI_MODEL=meta/llama-3.3-70b-instruct  # 模型名称

# 嵌入模型（本地，不需改）
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# 存储路径
VECTOR_STORE_PATH=./vector_store   # FAISS 索引位置
MEMORY_PATH=./memory               # 持久记忆 JSON 位置

# 参数调节
RETRIEVAL_TOP_K=4     # 每次检索返回几篇
MEMORY_WINDOW_SIZE=10 # 对话窗口轮数
MAX_TOKENS=2048       # 模型最大输出 token
```

---

## ⚠️ 当前状态（2026-04-17）

| 功能 | 状态 | 备注 |
|------|------|------|
| CLI 界面启动 | ✅ 正常 | Rich 美化界面 |
| PDF 加载 / 向量化 | ✅ 正常 | 本地 FastEmbed |
| bulk_search arXiv | ✅ 正常 | 实时联网 |
| 知识库 status | ✅ 正常 | - |
| 持久记忆 remember | ✅ 正常 | JSON 文件 |
| **LLM 问答（chat）** | ❌ 有 Bug | surrogate 编码错误见 README.md |
| survey 生成 | ❌ 依赖 chat | 同上 |
| arXiv Agent 搜索 | ❌ 依赖 chat | 同上 |

**核心问题：** `agent.chat()` 中 LangGraph 内部引入 `\udcaf` surrogate，导致 OpenAI SDK 编码 JSON 请求时崩溃。详见 `handoff/README.md` 第3节。

---

## 🚀 典型使用场景

### 场景一：读懂一篇论文
```
你: load D:\Attention_Is_All_You_Need.pdf
你: 这篇论文的核心贡献是什么？
你: Transformer 里的 Multi-Head Attention 怎么工作？
你: 它和 RNN 相比有什么优势？
```

### 场景二：调研一个领域
```
你: bulk_search RAG retrieval augmented generation survey 100
你: bulk_search RAG hallucination reduction 50
你: RAG 有哪些主流架构？各有什么优缺点？
你: survey retrieval augmented generation --out rag_survey.md
```

### 场景三：追踪最新进展
```
你: bulk_search multimodal LLM agent 2024 30
你: 2024 年以来多模态 LLM Agent 有哪些主要突破？
你: 有哪些值得精读的顶会论文？
you: remember 需要重点关注 ReAct 和 Reflexion 框架
```
