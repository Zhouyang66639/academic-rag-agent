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

console = Console()

BANNER = """
[bold cyan]
  +-----------------------------------------------+
  |       Academic RAG Agent  v1.0                |
  |   面向科研文献的记忆增强 LLM Agent            |
  |                                               |
  |   RAG + Memory + arXiv Search + LLM           |
  +-----------------------------------------------+
[/bold cyan]
"""

HELP_TEXT = """
[bold]可用命令:[/bold]
  [cyan]load <文件路径>[/cyan]     -- 加载 PDF/TXT/MD 文档到知识库
  [cyan]load_dir <目录路径>[/cyan]  -- 加载整个目录的文档
  [cyan]status[/cyan]             -- 查看当前知识库状态
  [cyan]clear_memory[/cyan]       -- 清空对话历史
  [cyan]clear_db[/cyan]           -- 清空知识库（慎用！）
  [cyan]help[/cyan]               -- 显示此帮助
  [cyan]exit / quit[/cyan]        -- 退出程序

[bold]示例问题:[/bold]
  这篇论文的核心贡献是什么？
  RAG 和 Fine-tuning 有什么区别？
  搜索一下 memory augmented LLM agent 相关论文
"""


def check_env():
    """检查必要的环境变量是否已配置"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-xxx") or api_key.startswith("AIzaxxx"):
        console.print(
            Panel(
                "[red]未配置 API Key！\n\n"
                "请复制 .env.example 为 .env 并填入你的 API Key:\n"
                "  copy .env.example .env\n"
                "  # 然后编辑 .env 文件",
                title="配置错误",
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
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    model_name = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
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
