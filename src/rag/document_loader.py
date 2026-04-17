"""
Document Loader Module
加载各种格式的学术文档（PDF、TXT、Markdown）
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console

console = Console()


class AcademicDocumentLoader:
    """
    学术文献加载器
    支持 PDF / TXT / Markdown 格式
    自动分块，适配 RAG 检索
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": "PDF 论文",
        ".txt": "文本文件",
        ".md": "Markdown 文档",
    }

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数（保证上下文连续性）
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文件并分块"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式: {ext}，"
                f"支持: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        console.print(
            f"[cyan]📄 加载文件:[/cyan] {path.name} "
            f"[dim]({self.SUPPORTED_EXTENSIONS[ext]})[/dim]"
        )

        # 根据扩展名选择加载器
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        elif ext == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(str(path))

        raw_docs = loader.load()

        # 添加元数据
        for doc in raw_docs:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = ext

        # 分块
        chunks = self.text_splitter.split_documents(raw_docs)
        console.print(
            f"[green]✅ 完成:[/green] 共生成 [bold]{len(chunks)}[/bold] 个文本块"
        )
        return chunks

    def load_directory(self, dir_path: str) -> List[Document]:
        """递归加载目录下的所有支持格式文档"""
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f"目录不存在: {dir_path}")

        all_docs = []
        files = [
            f for f in path.rglob("*")
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not files:
            console.print(f"[yellow]⚠️  目录中未找到支持的文件: {dir_path}[/yellow]")
            return []

        console.print(f"[blue]📂 扫描目录:[/blue] {dir_path} — 找到 {len(files)} 个文件")

        for f in files:
            try:
                docs = self.load_file(str(f))
                all_docs.extend(docs)
            except Exception as e:
                console.print(f"[red]❌ 加载失败:[/red] {f.name} — {e}")

        console.print(
            f"\n[bold green]📚 目录加载完成:[/bold green] "
            f"共 {len(files)} 个文件，{len(all_docs)} 个文本块\n"
        )
        return all_docs
