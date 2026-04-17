"""
Academic RAG Agent — 核心 Agent 模块

Architecture:
    用户问题
       ↓
    [Memory] 读取历史对话上下文
       ↓
    [Router] 判断是否需要检索本地文档 or 搜索 arXiv
       ↓
    [RAG Retriever] 从 FAISS 向量库检索相关段落
       ↓
    [LLM] 结合文档 + 历史 + 问题生成回答
       ↓
    [Memory] 存储本轮对话
"""

import os
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from rich.console import Console

from src.rag.vector_store import VectorStoreManager
from src.tools.arxiv_tool import arxiv_search_tool

console = Console()


SYSTEM_PROMPT = """你是一位专业的学术研究助手，擅长分析和讲解 AI 领域的学术论文，\
尤其专注于大语言模型（LLM）、RAG（检索增强生成）、AI Agent 等方向。

你拥有以下能力：
1. 📚 **文献问答**：从用户上传的论文中检索相关内容，精准回答问题
2. 🔍 **论文搜索**：在 arXiv 上搜索最新相关论文
3. 🧠 **记忆能力**：记住对话历史，支持多轮追问和上下文理解
4. 💡 **深度分析**：解释方法论、对比不同工作、总结研究贡献

回答要求：
- 回答要专业、简洁，适合研究者阅读
- 引用文档内容时，请注明来源（如：根据 [文件名] 第X页...）
- 如果文档中没有相关信息，诚实说明并建议搜索 arXiv
- 使用中文回答，技术术语保留英文

你的研究背景：熟悉 Transformer、Attention、RAG、Memory、Planning 等核心概念。"""


class AcademicRAGAgent:
    """
    学术文献 RAG Agent

    Integrates:
        - FAISS 向量检索（本地文档 RAG）
        - arXiv 实时搜索
        - ConversationBufferWindowMemory（滑动窗口对话记忆）
        - Tool-calling Agent（LangChain）
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        model_name: str = "deepseek-chat",
        memory_window: int = 10,
        retrieval_top_k: int = 4,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        self.vector_store = vector_store
        self.memory_window = memory_window

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # 初始化对话记忆
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
        )

        # 构建工具列表
        self.tools = self._build_tools(retrieval_top_k)

        # 构建 Agent
        self.agent_executor = self._build_agent()

        console.print(
            f"[bold green]🤖 Academic RAG Agent 初始化完成[/bold green]\n"
            f"   模型: {model_name} | 记忆窗口: {memory_window} 轮 | "
            f"检索 Top-K: {retrieval_top_k}"
        )

    def _build_tools(self, top_k: int) -> list:
        """构建 Agent 工具集"""
        tools = [arxiv_search_tool]

        # 如果向量库有内容，添加文档检索工具
        if self.vector_store.doc_count > 0:
            retriever = self.vector_store.get_retriever(k=top_k)
            doc_retrieval_tool = create_retriever_tool(
                retriever,
                name="search_uploaded_papers",
                description=(
                    "从用户上传的本地论文文档中检索相关内容。"
                    "当用户询问已上传文献的具体内容、方法、实验结果时使用此工具。"
                    "输入应为用户问题的核心语义内容。"
                ),
            )
            tools.append(doc_retrieval_tool)
            console.print(
                f"[cyan]📚 已加载文档检索工具[/cyan] "
                f"({self.vector_store.doc_count} 个向量块)"
            )

        return tools

    def _build_agent(self) -> AgentExecutor:
        """构建 Tool-calling Agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=False,          # 设为 True 可查看 Agent 推理过程
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def chat(self, user_input: str) -> str:
        """
        与 Agent 对话

        Args:
            user_input: 用户输入的问题

        Returns:
            Agent 的回答
        """
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response.get("output", "抱歉，我没能生成回答，请重试。")
        except Exception as e:
            return f"处理过程中发生错误: {e}"

    def reload_tools(self, top_k: int = 4):
        """重新加载工具（上传新文档后调用）"""
        self.tools = self._build_tools(top_k)
        self.agent_executor = self._build_agent()
        console.print("[green]🔄 工具已重新加载[/green]")

    def clear_memory(self):
        """清空对话历史"""
        self.memory.clear()
        console.print("[yellow]🧹 对话历史已清空[/yellow]")

    @property
    def history_turns(self) -> int:
        """返回当前记忆中的对话轮数"""
        messages = self.memory.chat_memory.messages
        return len(messages) // 2
