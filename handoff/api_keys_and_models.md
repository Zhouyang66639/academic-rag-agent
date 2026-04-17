# API 密钥 & 模型测试汇总

> 整理自 2026-04-17 本次调试会话

---

## 📦 API 提供商汇总

### 1. 硅基流动 (SiliconFlow)

| 项目 | 值 |
|------|-----|
| **API Key** | `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` |
| **Base URL** | `https://api.siliconflow.cn/v1` |
| **协议格式** | OpenAI 兼容 |
| **测试结论** | ❌ 未成功使用（模型 ID 不兼容，未深入测试） |

---

### 2. OpenRouter

| 项目 | 值 |
|------|-----|
| **API Key** | `sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` |
| **Base URL** | `https://openrouter.ai/api/v1` |
| **协议格式** | OpenAI 兼容 |
| **测试结论** | ⚠️ 免费模型限制多，Key 曾有消费限额（已手动解除） |

#### OpenRouter 模型测试结果

| 模型 ID | 简单对话 | 工具调用 | 失败原因 |
|---------|---------|---------|---------|
| `meta-llama/llama-3.3-70b-instruct:free` | ❌ | ❌ | 429 频繁限流 |
| `google/gemma-3-27b-it:free` | ✅ 干净 | ❌ | 不支持 Tool Calling（404） |
| `google/gemma-4-26b-a4b-it:free` | ✅ | ✅ | 可用，但需看负载 |
| `google/gemma-4-31b-it:free` | ✅ | ✅ | Full Agent 下超时 |
| `openai/gpt-oss-120b:free` | ✅ | ✅ | 可用 |
| `openai/gpt-oss-20b:free` | ✅ | ✅ | 可用 |
| `z-ai/glm-4.5-air:free` | ✅ | ✅ | 可用 |
| `qwen/qwen3-coder:free` | ❌ | ❌ | 429 限流 |
| `qwen/qwen3-next-80b-a3b-instruct:free` | ❌ | ❌ | 429 限流 |
| `mistralai/mistral-7b-instruct:free` | ❌ | ❌ | 404 不存在 |
| `microsoft/phi-4-reasoning:free` | ❌ | ❌ | 404 不存在 |
| `deepseek/deepseek-r1-distill-llama-70b:free` | ❌ | ❌ | 404 不存在 |

> **注意：** OpenRouter 免费模型的 429 错误响应体本身含非 UTF-8 字节，会触发 Windows surrogate 崩溃（已修复）

---

### 3. 幻城网安 (iamhc.cn) ← 当前使用

| 项目 | 值 |
|------|-----|
| **API Key** | `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` |
| **Base URL** | `https://api.iamhc.cn/v1` |
| **协议格式** | OpenAI 兼容 |
| **测试结论** | ✅ 连通性好，模型品种多，工具调用正常 |
| **官网推荐** | nvidia/nemotron-3-super-120b-a12b、gpt-120b-oss、deepseek v3.1、kimi k2.5、glm4.7、glm5、qwen3.5、mimmax2.5 |

#### iamhc.cn 模型测试结果

| 模型 ID | 简单对话 | 工具调用 | 备注 |
|---------|---------|---------|------|
| **`meta/llama-3.3-70b-instruct`** | ✅ | **✅ 正确** | **当前 .env 使用，推荐** |
| **`gpt-5.4`** | ✅ | **✅ 正确** | 备选，未知底层模型 |
| `meta/llama-4-maverick-17b-128e-instruct` | ✅ | ⚠️ 格式错误 | 把工具调用当文本输出，不兼容 LangChain |
| `deepseek-ai/deepseek-v3.2` | ❌ | ❌ | 超时（服务器高负载） |
| `deepseek-ai/deepseek-v3.1` | ❌ | ❌ | 410 已下线 |
| `glm-5.1` | ❌ | ❌ | 429 配额耗尽 |
| `glm-5` | ❌ | ❌ | 429 配额耗尽 |
| `google/gemma-4-31b-it` | ❌ | ❌ | 超时 |

> **iamhc.cn 完整模型列表（从 /v1/models 获取，2026-04-17）包含 80+ 模型，**  
> 含 deepseek 系列、glm 系列、llama 系列、gemma 系列、Qwen 系列等，多数未测试。

---

## 🏆 最终推荐配置

```env
# 首选（当前使用）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.iamhc.cn/v1
OPENAI_MODEL=meta/llama-3.3-70b-instruct

# 备选 1（同一 API，工具调用也 OK）
OPENAI_MODEL=gpt-5.4

# 备选 2（OpenRouter，高峰期可能限流）
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-oss-120b:free
```

---

## 🔑 换 API 操作步骤

1. 编辑 `.env` 文件，修改三行：
   ```
   OPENAI_API_KEY=新key
   OPENAI_BASE_URL=新地址/v1
   OPENAI_MODEL=模型ID
   ```
2. 用 `check_models.py` 验证新配置：
   ```bash
   python check_models.py
   ```
3. 启动主程序：
   ```bash
   python main.py
   ```

---

## ⚙️ 工具调用兼容性说明

本 Agent 使用 LangGraph ReAct 模式，**必须支持原生 Function Calling**。

- ✅ 支持：Llama 3.x、GPT 系列、GLM-4 系列（需测试）、Gemma 4
- ❌ 不支持：Gemma 3（OpenRouter）、部分 Llama 4 变体（格式不兼容）
- 🔍 判断方法：用 `check_models.py` 脚本中的工具调用测试
