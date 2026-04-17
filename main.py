"""
Main Entry Point - 交互式命令行界面
运行方式: python main.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

# 加载环境变量（绝对路径，从任意目录都能找到 .env）
_PROJECT_ROOT = Path(__file__).parent
load_dotenv(_PROJECT_ROOT / ".env")

from src.rag.document_loader import AcademicDocumentLoader
from src.rag.vector_store import VectorStoreManager
from src.agent.agent import AcademicRAGAgent
from src.tools.bulk_arxiv import bulk_fetch_arxiv
from src.tools.survey_writer import generate_survey
from src.memory.persistent_memory import PersistentMemory

console = Console()

BANNER = """
[bold cyan]
  +--------------------------------------------------+
  |       Academic RAG Agent  v2.0                   |
  |   Memory-Augmented LLM Agent for Research        |
  |                                                  |
  |   LangGraph + Hybrid Search + Survey Writer      |
  +--------------------------------------------------+
[/bold cyan]
"""

HELP_TEXT = """
[bold]Document Commands:[/bold]
  [cyan]load <path>[/cyan]              -- Index a PDF / TXT / MD file
  [cyan]load_dir <directory>[/cyan]     -- Index all documents in a folder

[bold]arXiv Commands:[/bold]
  [cyan]bulk_search <topic> [N][/cyan]  -- Fetch & index N abstracts (default 50)
                               Example: bulk_search RAG LLM 100

[bold]Survey Generation:[/bold]
  [cyan]survey <topic>[/cyan]           -- Generate a full survey paper (Markdown)
  [cyan]survey <topic> --out <file>[/cyan]  -- Save to specific file

[bold]Memory Commands:[/bold]
  [cyan]remember <note>[/cyan]          -- Save a fact/note to permanent memory
  [cyan]memories[/cyan]                 -- Show all saved memories & session archive
  [cyan]forget <id>[/cyan]              -- Delete a memory by ID
  [cyan]clear_memory[/cyan]             -- Archive this session and reset chat

[bold]System Commands:[/bold]
  [cyan]status[/cyan]                   -- Show knowledge base & memory stats
  [cyan]clear_db[/cyan]                 -- Wipe the vector store
  [cyan]help[/cyan]                     -- Show this help
  [cyan]exit[/cyan]                     -- Save session and quit

[bold]Example questions:[/bold]
  What is the core contribution of the Attention paper?
  How does RAG differ from fine-tuning?
  Search for recent papers on LLM agents
"""


def check_env():
    """Check required environment variables are configured."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    placeholders = ("sk-xxx", "AIzaxxx", "gsk_xxx", "your_key", "your-key")
    if not api_key or any(api_key.startswith(p) for p in placeholders):
        console.print(
            Panel(
                "[red]API Key not configured!\n\n"
                "Copy .env.example to .env and fill in your API key:\n"
                "  copy .env.example .env",
                title="Configuration Error",
                border_style="red",
            )
        )
        sys.exit(1)


def print_status(vector_store: VectorStoreManager, agent: AcademicRAGAgent):
    """Print current system status including memory stats."""
    pm = agent.persistent_memory
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Item", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Knowledge base", f"[bold]{vector_store.doc_count}[/bold] vectors")
    table.add_row("Chat turns (session)", f"[bold]{agent.history_turns}[/bold]")
    table.add_row("Fact memory", f"[bold]{pm.fact_count}[/bold] saved notes")
    table.add_row("Session archive", f"[bold]{pm.session_count}[/bold] sessions")
    table.add_row("Vector store path", str(vector_store.store_path))
    table.add_row("Memory path", str(pm.memory_dir))
    console.print(Panel(table, title="System Status", border_style="blue"))


def run_interactive(agent: AcademicRAGAgent, loader: AcademicDocumentLoader, vector_store: VectorStoreManager):
    """运行交互式对话循环"""
    console.print(Panel(HELP_TEXT, title="使用指南", border_style="dim"))

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]你[/bold green]").strip()

            if not user_input:
                continue

            # -- Commands --
            if user_input.lower() in ("exit", "quit", "q"):
                # Auto-save session summary before exiting
                if agent.history_turns > 0:
                    console.print("[dim]Saving session summary...[/dim]")
                    try:
                        summary = agent.chat(
                            "Summarise our conversation in 1-2 sentences for future reference."
                        )
                        agent.persistent_memory.save_session(summary, agent.history_turns)
                    except Exception:
                        pass
                console.print("[dim]Goodbye![/dim]")
                break

            elif user_input.lower() == "help":
                console.print(Panel(HELP_TEXT, title="帮助", border_style="dim"))

            elif user_input.lower() == "status":
                print_status(vector_store, agent)

            elif user_input.lower() == "clear_memory":
                # Auto-generate summary before clearing
                summary = ""
                if agent.history_turns > 0:
                    try:
                        summary = agent.chat(
                            "In 1-2 sentences, summarise what we discussed this session."
                        )
                    except Exception:
                        pass
                agent.clear_memory(summary=summary)

            elif user_input.lower() == "clear_db":
                confirm = Prompt.ask(
                    "[red]确认清空知识库？这将删除所有已索引文档（输入 YES 确认）[/red]"
                )
                if confirm == "YES":
                    vector_store.clear()
                    agent.reload_tools()

            elif user_input.lower().startswith("load_dir "):
                dir_path = user_input[9:].strip().strip('"')
                docs = loader.load_directory(dir_path)
                if docs:
                    vector_store.add_documents(docs)
                    agent.reload_tools()

            elif user_input.lower().startswith("load "):
                file_path = user_input[5:].strip().strip('"')
                docs = loader.load_file(file_path)
                if docs:
                    vector_store.add_documents(docs)
                    agent.reload_tools()

            elif user_input.lower().startswith("remember "):
                note = user_input[9:].strip()
                agent.persistent_memory.remember(note)

            elif user_input.lower() == "memories":
                agent.persistent_memory.show_memories()

            elif user_input.lower().startswith("forget "):
                try:
                    fact_id = int(user_input[7:].strip())
                    agent.persistent_memory.forget(fact_id)
                except ValueError:
                    console.print("[red]Usage: forget <id number>[/red]")

            elif user_input.lower().startswith("bulk_search "):
                # bulk_search <topic> [max_results]
                # e.g.: bulk_search RAG retrieval augmented generation 100
                parts = user_input[12:].strip()
                # Check if last token is a number
                tokens = parts.split()
                if tokens and tokens[-1].isdigit():
                    max_n = int(tokens[-1])
                    topic = " ".join(tokens[:-1])
                else:
                    max_n = 50
                    topic = parts
                console.print(
                    f"[cyan]Bulk indexing up to {max_n} arXiv papers on:[/cyan] {topic}"
                )
                docs = bulk_fetch_arxiv(topic, max_results=max_n)
                if docs:
                    vector_store.add_documents(docs)
                    agent.reload_tools()
                    console.print(
                        f"[green]{len(docs)} papers indexed. You can now ask questions across all of them![/green]"
                    )

            elif user_input.lower().startswith("survey "):
                # survey <topic> [--out filename.md]
                parts = user_input[7:].strip()
                out_file = None
                if "--out " in parts:
                    idx = parts.index("--out ")
                    out_file = parts[idx + 6:].strip()
                    topic = parts[:idx].strip()
                else:
                    topic = parts

                if vector_store.doc_count == 0:
                    console.print(
                        "[yellow]No documents indexed yet.\n"
                        "Tip: run 'bulk_search <topic> 50' first to index papers![/yellow]"
                    )
                else:
                    retriever = vector_store.get_hybrid_retriever(k=5)
                    _, saved_path = generate_survey(
                        topic=topic,
                        retriever=retriever,
                        llm=agent.llm,
                        output_path=out_file,
                    )
                    console.print(f"[bold green]Survey saved to:[/bold green] {saved_path}")

            # 正常对话
            else:
                console.print("\n[bold blue]Agent[/bold blue] [dim]正在思考...[/dim]")
                response = agent.chat(user_input)
                console.print(
                    Panel(
                        response,
                        title=f"[bold blue]Agent[/bold blue] [dim](历史: {agent.history_turns} 轮)[/dim]",
                        border_style="blue",
                        padding=(1, 2),
                    )
                )

        except KeyboardInterrupt:
            console.print("\n[dim]按 Ctrl+C 再次退出，或输入 exit[/dim]")
        except EOFError:
            break


def main():
    console.print(BANNER)
    check_env()

    # 初始化组件
    store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    model_name = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    memory_window = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    max_tokens = int(os.getenv("MAX_TOKENS", "2048"))

    console.print("[dim]正在初始化系统...[/dim]\n")

    memory_dir = os.getenv("MEMORY_PATH", "./memory")

    persistent_memory = PersistentMemory(memory_dir=memory_dir)
    vector_store = VectorStoreManager(store_path, embedding_model)
    loader = AcademicDocumentLoader()
    agent = AcademicRAGAgent(
        vector_store=vector_store,
        model_name=model_name,
        memory_window=memory_window,
        retrieval_top_k=top_k,
        max_tokens=max_tokens,
        persistent_memory=persistent_memory,
    )

    run_interactive(agent, loader, vector_store)


if __name__ == "__main__":
    main()
