# Academic RAG Agent — AI 交接文档

> 生成时间：2026-04-17  
> 目的：节省 token，让下一个 AI 快速恢复上下文

---

## 目录

1. [项目概述](#1-项目概述)
2. [当前配置（已确认可用）](#2-当前配置)
3. [未解决的核心 Bug](#3-未解决的核心-bug)
4. [已修改的文件清单](#4-已修改的文件清单)
5. [关键代码片段](#5-关键代码片段)
6. [诊断历程（避免重复踩坑）](#6-诊断历程)
7. [下一步任务](#7-下一步任务)
8. [启动方式](#8-启动方式)

---

## 1. 项目概述

**项目路径：** `d:\python study\ai_coding\academic-rag-agent`  
**启动命令：** `conda activate rag-agent && python main.py`  
**Python 环境：** `C:\Users\16273\miniconda3\envs\rag-agent\python.exe`

### 架构
```
用户输入
  ↓
main.py (Rich CLI 交互层)
  ↓
AcademicRAGAgent.chat() → LangGraph create_react_agent (ReAct 循环)
  ├─ Tool: arxiv_search_tool (实时 arXiv API)
  └─ Tool: search_uploaded_papers (BM25+FAISS+RRF 混合检索)
  ↓
MemorySaver (per-thread 对话记忆)
  ↓
PersistentMemory (跨会话记忆，JSON 文件)
```

---

## 2. 当前配置

### `.env` 文件（当前值）

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.iamhc.cn/v1
OPENAI_MODEL=meta/llama-3.3-70b-instruct

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
VECTOR_STORE_PATH=./vector_store
MEMORY_PATH=./memory
RETRIEVAL_TOP_K=4
MEMORY_WINDOW_SIZE=10
MAX_TOKENS=2048
```

### API 提供商说明（iamhc.cn）

- **地址：** `https://api.iamhc.cn/v1`（OpenAI 兼容格式）
- **官网推荐模型：** nvidia/nemotron-3-super-120b-a12b、gpt-120b-oss、deepseek v3.1、kimi k2.5、glm4.7、glm5、qwen3.5、mimmax2.5
- **实测可用的模型 ID（在此 API 上）：**

| 模型 ID | 工具调用 | 状态 |
|---------|---------|------|
| `meta/llama-3.3-70b-instruct` | ✅ 正确 | 当前使用，**BUT 有 surrogate bug（见第3节）** |
| `gpt-5.4` | ✅ 正确 | 可用备选 |
| `deepseek-ai/deepseek-v3.2` | ⚠️ 超时 | 高负载 |
| `glm-5.1` / `glm-5` | ❌ 429 | 限额 |

---

## 3. 未解决的核心 Bug ⚠️

**✅ 程序不崩溃**（有保护层），**但每次聊天都返回 Error 而非真实回复。**

### 错误现象
```
Error: 'utf-8' codec can't encode character '\udcaf' in position 1942: surrogates not allowed
```

### 完整 Traceback（已确认）
```
src/agent/agent.py → graph.invoke()
  → langgraph/pregel/main.py
  → langchain_openai/chat_models/base.py line 1498: client.with_raw_response.create(**payload)
  → openai/_base_client.py line 568: openapi_dumps(json_data).encode()
  → openai/_utils/_json.py line 25: .encode()
UnicodeEncodeError: 'utf-8' codec can't encode character '\udcaf' in position 1942
```

### 根本原因（已诊断，未修复）

`openapi_dumps` 使用 `json.dumps(obj, ensure_ascii=False).encode()` 对请求 payload 编码。  
position 1942 处有一个 lone surrogate `\udcaf`。

**已排除**的来源：
- ❌ 静态系统提示（SYSTEM_PROMPT）— 测试确认无 surrogate
- ❌ 工具定义（tool specs）— 测试确认无 surrogate  
- ❌ arXiv 工具返回值 — 已加 `_clean()` 函数

**疑似来源（未验证）：**  
LangGraph 的 `create_react_agent` 在 `prompt=` 参数传入字符串时，内部会构建 `ChatPromptTemplate`。该模板在运行时 `format_messages()` 阶段可能从某处引入 surrogate（可能是 `→` U+2192 被某个旧版 LangChain 的 Jinja2 模板引擎错误处理）。

### 建议下一步排查

**选项 A（推荐）：** 调整 `_build_graph()` 里的 prompt 传参方式

```python
# 当前写法（可能有问题）：
return create_react_agent(
    model=self.llm,
    tools=self.tools,
    prompt=prompt,           # <-- 字符串
    checkpointer=self.checkpointer,
)

# 尝试改为 SystemMessage 对象：
from langchain_core.messages import SystemMessage
return create_react_agent(
    model=self.llm,
    tools=self.tools,
    prompt=SystemMessage(content=prompt),  # <-- Message 对象
    checkpointer=self.checkpointer,
)
```

**选项 B：** 用 `state_modifier` 替代 `prompt=`

```python
from langchain_core.messages import SystemMessage

def add_system(state):
    return [SystemMessage(content=prompt)] + state["messages"]

return create_react_agent(
    model=self.llm,
    tools=self.tools,
    state_modifier=add_system,
    checkpointer=self.checkpointer,
)
```

**选项 C：** 升级/降级 langchain-core 版本

当前安装版本（从 traceback 推断）：`langchain-core >= 0.3`。可以尝试：
```bash
pip install langchain-core==0.2.43 langgraph==0.1.17
```

**选项 D：** 对 SYSTEM_PROMPT 做 ASCII-safe 处理（治标）

```python
# 在 _build_graph() 里：
prompt_safe = SYSTEM_PROMPT.encode('ascii', errors='xmlcharrefreplace').decode('ascii')
```

---

## 4. 已修改的文件清单

| 文件 | 修改内容 |
|------|---------|
| `main.py` | 顶部加 Windows UTF-8 encoding 强制设置；响应显示加双层 try/except |
| `src/agent/agent.py` | `streaming=False`；`chat()` 加 surrogate 清洗；加 `traceback.print_exc()`；系统提示加"When NOT to use tools"；`recursion_limit=10` |
| `src/tools/arxiv_tool.py` | 加 `_clean()` 函数；对所有 arXiv 返回字段清洗 surrogate |
| `.env` | API Key/Base/Model 三次变更（Gemini → SiliconFlow → OpenRouter → iamhc.cn） |
| `check_models.py` | 新建，用于快速测试不同模型的可用性和工具调用支持 |
| `diagnose.py` | 新建，用于逐层诊断 surrogate 来源（调试用，可删除） |

---

## 5. 关键代码片段

### main.py — Windows UTF-8 修复
```python
import sys, io, os
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8:replace')
```

### main.py — 响应显示双层保护
```python
# 正常对话
else:
    console.print("\n[bold blue]Agent[/bold blue] [dim]正在思考...[/dim]")
    try:
        response = agent.chat(user_input)
        response = response.encode("utf-8", errors="replace").decode("utf-8")
    except Exception as e:
        response = str(e).encode("utf-8", errors="replace").decode("utf-8")
        response = f"Error: {response}"
    console.print(Panel(response, ...))
```

### agent.py — chat() 方法（当前完整版）
```python
def chat(self, user_input: str) -> str:
    try:
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={**self._config, "recursion_limit": 10},
        )
        self._turn_count += 1
        raw = result["messages"][-1].content
        return raw.encode("utf-8", errors="replace").decode("utf-8")
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_str = str(e).encode("utf-8", errors="replace").decode("utf-8")
        return f"Error: {err_str}"
```

### arxiv_tool.py — surrogate 清洗
```python
def _clean(text: str) -> str:
    """Remove lone surrogates from text."""
    return text.encode("utf-8", errors="replace").decode("utf-8")

@tool
def arxiv_search_tool(query: str) -> str:
    ...
    results = search_arxiv(query, max_results=5)
    for paper in results:
        paper["title"] = _clean(paper["title"])
        paper["abstract"] = _clean(paper["abstract"])
        paper["authors"] = [_clean(a) for a in paper["authors"]]
    output = format_arxiv_results(results)
    return _clean(output)
```

---

## 6. 诊断历程（避免重复踩坑）

### 问题演化时间线

1. **最初问题**：OpenRouter Key 有消费限额，返回 402 错误。→ 修复：去 openrouter.ai/keys 取消限额
2. **402→429**：OpenRouter 免费模型 (llama-3.3-70b) 高峰期 429 限流。→ 尝试多模型
3. **Gemma 3 27B**：`google/gemma-3-27b-it:free` 可连通但**不支持工具调用**（404）。
4. **Gemma 4**：`google/gemma-4-31b-it:free` 支持工具调用，但在完整 agent 下 429。
5. **换 API**：切到 `api.iamhc.cn`，`meta/llama-3.3-70b-instruct` 工具调用测试 ✅。
6. **工具循环 Bug**：对 "hello" 无限调用 arXiv。→ 修复：系统提示加"When NOT to use tools"，加 `recursion_limit=10`。
7. **当前问题**：position 1942 UnicodeEncodeError 在 `openai._utils._json.openapi_dumps().encode()`。
   - 静态 payload（system prompt + tools + user msg）测试无 surrogate ✅
   - 结论：surrogate 由 LangGraph 运行时动态引入，怀疑在 `prompt=` 参数处理

### 关键发现
- **OpenRouter 免费模型的 surrogate** 来自 429 error response 的 raw bytes（已修复）
- **iamhc.cn 的 surrogate** 来自 LangGraph 内部，与 API 无关
- `position 1543` / `position 1942` 随系统提示长度变化（确认 surrogate 在 payload 的固定位置）

---

## 7. 下一步任务

### 优先级 P0（阻塞性）
- [ ] **修复 surrogate bug**：尝试上述[选项 A/B/C/D](#建议下一步排查)中的一个

### 优先级 P1（功能完善）
- [ ] 测试 RAG 功能：`load <pdf路径>` 后问相关问题，验证检索是否正确
- [ ] 测试 arXiv 搜索：`bulk_search RAG LLM 20` 后提问
- [ ] 测试 survey 生成：`survey retrieval augmented generation`

### 优先级 P2（工程优化）
- [ ] 删除调试文件：`diagnose.py`（纯诊断用）
- [ ] 加指数退避重试（tenacity）应对 API 限流
- [ ] 集成 BGE-Reranker
- [ ] 开发 Streamlit WebUI

---

## 8. 启动方式

```powershell
# 在 Conda rag-agent 环境下
cd "d:\python study\ai_coding\academic-rag-agent"
conda activate rag-agent
python main.py

# 快速测试 API 连通性
python check_models.py

# 查看错误的完整 traceback（stderr 单独输出）
python main.py 2> err.log
# 然后查看 err.log
```

### 常见命令（程序内）
```
你: status              # 查看知识库状态
你: load D:\path\to.pdf # 加载 PDF
你: bulk_search RAG 20  # 获取 20 篇 arXiv 论文
你: What is RAG?        # 正常提问
你: survey RAG LLM      # 生成综述
```
