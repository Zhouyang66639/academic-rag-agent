"""
Academic RAG Agent -- 核心 Agent 模块（ReAct 模式，兼容所有模型）

Architecture:
    用户问题
       |
    [Memory] 读取历史对话上下文
       |
    [ReAct Agent] LLM 文字推理 -> 决定用哪个工具 -> 执行工具 -> 生成回答
       |
    [Memory] 存储本轮对话
"""

import os
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools.retriever import create_retriever_tool
from rich.console import Console

from src.rag.vector_store import VectorStoreManager
from src.tools.arxiv_tool import arxiv_search_tool

console = Console()


class AcademicRAGAgent:
    """
    学术文献 RAG Agent（ReAct 模式）

    使用 ReAct (Reasoning + Acting) 模式驱动工具调用，
    通过文字推理完成工具选择，兼容 7B 小模型。

    Integrates:
        - FAISS 向量检索（本地文档 RAG）
        - arXiv 实时搜索
        - ConversationBufferWindowMemory（滑动窗口对话记忆）
        - ReAct Agent（LangChain，无需 Function Calling API）
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
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

        # 初始化对话记忆（滑动窗口）
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
        )

        # 构建工具列表
        self.tools = self._build_tools(retrieval_top_k)

        # 构建 ReAct Agent
        self.agent_executor = self._build_agent()

        console.print(
            f"[bold green]Academic RAG Agent 初始化完成[/bold green]\n"
            f"  模型: {model_name} | 记忆: {memory_window} 轮 | "
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
                    "当用户询问已上传文献的具体内容、方法或实验结果时使用此工具。"
                    "输入应为用户问题的核心关键词。"
                ),
            )
            tools.append(doc_retrieval_tool)
            console.print(
                f"[cyan]文档检索工具已加载[/cyan] "
                f"({self.vector_store.doc_count} 个向量块)"
            )

        return tools

    def _build_agent(self):
        """构建 ReAct Agent（兼容所有指令跟随模型）"""
        agent_kwargs = {
            "system_message": (
                "你是一位专业的学术研究助手，专注于 AI 领域（LLM、RAG、Agent 等）。\n"
                "你有以下工具可以使用：\n"
                "- arxiv_search_tool: 搜索 arXiv 上最新的学术论文\n"
                "- search_uploaded_papers: 从用户上传的本地文档中检索内容\n\n"
                "回答要求：专业简洁，使用中文，技术术语保留英文。\n"
                "引用文档内容时注明来源。"
            )
        }

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
            agent_kwargs=agent_kwargs,
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
        console.print("[green]工具已重新加载[/green]")

    def clear_memory(self):
        """清空对话历史"""
        self.memory.clear()
        console.print("[yellow]对话历史已清空[/yellow]")

    @property
    def history_turns(self) -> int:
        """返回当前记忆中的对话轮数"""
        messages = self.memory.chat_memory.messages
        return len(messages) // 2
