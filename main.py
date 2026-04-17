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

# 加载环境变量
load_dotenv()

from src.rag.document_loader import AcademicDocumentLoader
from src.rag.vector_store import VectorStoreManager
from src.agent.agent import AcademicRAGAgent
from src.tools.bulk_arxiv import bulk_fetch_arxiv
from src.tools.survey_writer import generate_survey

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
  [cyan]bulk_search <topic> [N][/cyan]  -- Fetch & index N abstracts from arXiv (default 50)
                               Example: bulk_search RAG large language model 100

[bold]Survey Generation:[/bold]
  [cyan]survey <topic>[/cyan]           -- Generate a full survey paper (Markdown)
                               Example: survey Retrieval-Augmented Generation
  [cyan]survey <topic> --out <file>[/cyan]  -- Save to specific file

[bold]System Commands:[/bold]
  [cyan]status[/cyan]                   -- Show knowledge base stats
  [cyan]clear_memory[/cyan]             -- Reset conversation history
  [cyan]clear_db[/cyan]                 -- Wipe the vector store
  [cyan]help[/cyan]                     -- Show this help
  [cyan]exit[/cyan]                     -- Quit

[bold]Example questions:[/bold]
  What is the core contribution of the Attention paper?
  How does RAG differ from fine-tuning?
  Compare the methods in the indexed papers
"""


def check_env():
    """Check required environment variables are configured."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    placeholders = ("sk-xxx", "AIzaxxx", "gsk_xxx", "your_key", "")
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
    """打印当前系统状态"""
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("项目", style="cyan")
    table.add_column("状态", style="white")
    table.add_row("知识库向量数", f"[bold]{vector_store.doc_count}[/bold] 个块")
    table.add_row("对话历史轮数", f"[bold]{agent.history_turns}[/bold] 轮")
    table.add_row("记忆窗口大小", f"{agent.memory_window} 轮")
    table.add_row("向量库路径", str(vector_store.store_path))
    console.print(Panel(table, title="系统状态", border_style="blue"))


def run_interactive(agent: AcademicRAGAgent, loader: AcademicDocumentLoader, vector_store: VectorStoreManager):
    """运行交互式对话循环"""
    console.print(Panel(HELP_TEXT, title="使用指南", border_style="dim"))

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]你[/bold green]").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]再见！[/dim]")
                break

            elif user_input.lower() == "help":
                console.print(Panel(HELP_TEXT, title="帮助", border_style="dim"))

            elif user_input.lower() == "status":
                print_status(vector_store, agent)

            elif user_input.lower() == "clear_memory":
                agent.clear_memory()

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

    vector_store = VectorStoreManager(store_path, embedding_model)
    loader = AcademicDocumentLoader()
    agent = AcademicRAGAgent(
        vector_store=vector_store,
        model_name=model_name,
        memory_window=memory_window,
        retrieval_top_k=top_k,
        max_tokens=max_tokens,
    )

    run_interactive(agent, loader, vector_store)


if __name__ == "__main__":
    main()
