"""
Academic RAG Agent — LangGraph ReAct Agent (v2.0)

Replaces the deprecated LangChain AgentExecutor with LangGraph's
prebuilt create_react_agent, which is the current industry standard
for production LLM agent systems (as of 2024-2025).

Key upgrades over v1:
    - LangGraph StateGraph replaces AgentExecutor
    - MemorySaver checkpointing replaces ConversationBufferWindowMemory
    - Hybrid BM25+FAISS retriever replaces dense-only FAISS
    - Thread-based session management (supports multi-user extension)

Architecture:
    User Input
        |
    LangGraph StateGraph (ReAct loop)
        |-- Tool: search_uploaded_papers (BM25 + FAISS + RRF)
        |-- Tool: arxiv_search_tool (live arXiv API)
        |
    MemorySaver (per-thread checkpointing, in-memory)
        |
    AI Response
"""

import os
import uuid
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from src.rag.vector_store import VectorStoreManager
from src.tools.arxiv_tool import arxiv_search_tool
from src.memory.persistent_memory import PersistentMemory

console = Console()

SYSTEM_PROMPT = """You are a professional academic research assistant specializing in AI literature \
(LLM, RAG, Agents, Transformers, Diffusion Models, etc.).

## Tools available
- search_uploaded_papers : search locally indexed PDFs and arXiv abstracts via Hybrid BM25+FAISS+RRF
- arxiv_search_tool      : search arXiv for the latest papers (live API)

## CRITICAL — Query Language Rule
**ALL tool queries MUST be in English**, regardless of the user's input language.
If the user asks in Chinese, silently translate the core concepts to English before calling any tool.
Example: "RAG的最新进展" → query tool with "recent advances retrieval augmented generation"

## Reasoning framework (Chain-of-Thought)
1. Understand what the user needs (decompose complex questions)
2. Decide which tool(s) to use — prefer local search first, then arXiv
3. Execute tool calls with precise English queries
4. Synthesise results critically — note agreements, contradictions, gaps
5. Respond in Chinese with technical terms kept in English

## Citation guidelines
- Always cite the source: [Paper Title, arxiv:XXXX] or [filename.pdf, p.N]
- If multiple papers agree, note the consensus
- Flag conflicting findings explicitly

## Quality standards
- Be technically precise; prefer specific numbers and methods over vague descriptions
- If retrieved context is insufficient, say so and suggest what to search next
- Proactively use arxiv_search_tool when local docs lack recent papers (≥2023)
"""


class AcademicRAGAgent:
    """
    LangGraph-based Academic RAG Agent.

    Uses create_react_agent (LangGraph prebuilt) with:
        - MemorySaver: thread-scoped in-process checkpointing
        - Hybrid retriever: BM25 (sparse) + FAISS MMR (dense) + RRF fusion
        - arXiv tool: real-time paper search

    Thread model:
        Each session gets a UUID thread_id. MemorySaver persists the full
        message history per thread. clear_memory() starts a new thread.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        memory_window: int = 10,         # kept for API compat, LangGraph handles internally
        retrieval_top_k: int = 4,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        persistent_memory: PersistentMemory = None,
    ):
        self.vector_store = vector_store
        self.memory_window = memory_window
        self._top_k = retrieval_top_k
        self._turn_count = 0

        # Persistent memory (facts + session archive)
        self.persistent_memory = persistent_memory or PersistentMemory()

        # Restore thread_id from last session, or start fresh
        saved_thread = self.persistent_memory.thread_id
        self._thread_id = saved_thread if saved_thread else str(uuid.uuid4())
        self.persistent_memory.save_thread_id(self._thread_id)

        # LLM (OpenAI-compatible endpoint)
        self.llm = ChatOpenAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # MemorySaver: stateful conversation checkpointing per thread
        self.checkpointer = MemorySaver()

        # Build tools and compile graph
        self.tools = self._build_tools(retrieval_top_k)
        self.graph = self._build_graph()

        console.print(
            f"[bold green]Academic RAG Agent v2.0 (LangGraph) ready[/bold green]\n"
            f"  Model: {model_name}\n"
            f"  Retriever: Hybrid (BM25 + FAISS + RRF)\n"
            f"  Memory: LangGraph MemorySaver | Session: {self._thread_id[:8]}..."
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_tools(self, top_k: int) -> list:
        """Build tool list: arXiv + optional hybrid retriever.

        Uses MultiQueryRetriever (Ma et al., 2023) when docs are available —
        generates 3 query variants per search to improve recall by 20-40%.
        Falls back to standard hybrid if MultiQueryRetriever is unavailable.
        """
        tools = [arxiv_search_tool]

        # Prefer MultiQueryRetriever → Hybrid → Dense (in that order)
        retriever = self.vector_store.get_multi_query_retriever(
            llm=self.llm, k=top_k
        )
        if retriever is not None:
            retriever_tool = create_retriever_tool(
                retriever,
                name="search_uploaded_papers",
                description=(
                    "Search locally indexed research documents (PDF/TXT/MD/arXiv). "
                    "Uses Multi-Query + Hybrid BM25+FAISS+RRF retrieval. "
                    "Input MUST be English keywords or a full English question."
                ),
            )
            tools.append(retriever_tool)

        return tools

    def _build_graph(self):
        """
        Compile LangGraph ReAct agent.

        Injects persistent memory context (saved facts + session summaries)
        into the system prompt so the agent 'remembers' across restarts.
        """
        mem_context = self.persistent_memory.build_context()
        prompt = SYSTEM_PROMPT + ("\n\n" + mem_context if mem_context else "")

        return create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=prompt,
            checkpointer=self.checkpointer,
        )

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    @property
    def _config(self) -> dict:
        """LangGraph thread config — same thread_id = continuous memory."""
        return {"configurable": {"thread_id": self._thread_id}}

    def chat(self, user_input: str) -> str:
        """
        Send a message; memory is maintained automatically via LangGraph state.

        LangGraph appends HumanMessage → runs ReAct loop → returns AIMessage.
        The full message history lives in MemorySaver keyed by thread_id.
        """
        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=self._config,
            )
            self._turn_count += 1
            # Last element in messages list is always the final AI response
            return result["messages"][-1].content
        except Exception as e:
            return f"Error: {e}"

    def reload_tools(self, top_k: Optional[int] = None):
        """Reload tools after new documents are indexed."""
        top_k = top_k or self._top_k
        self.tools = self._build_tools(top_k)
        self.graph = self._build_graph()
        console.print("[green]Agent graph rebuilt with updated tools[/green]")

    def clear_memory(self, summary: str = ""):
        """
        Clear conversation memory by starting a new LangGraph thread.

        If a summary is provided, archives it to episodic memory before clearing.
        Previous thread data remains in MemorySaver but is no longer referenced.
        """
        # Archive current session to episodic memory
        if summary and self._turn_count > 0:
            self.persistent_memory.save_session(summary, self._turn_count)

        self._thread_id = str(uuid.uuid4())
        self.persistent_memory.save_thread_id(self._thread_id)
        self._turn_count = 0
        # Rebuild graph to inject latest memory context into prompt
        self.graph = self._build_graph()
        console.print(
            f"[yellow]Memory cleared — new session: {self._thread_id[:8]}...[/yellow]"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history_turns(self) -> int:
        return self._turn_count
