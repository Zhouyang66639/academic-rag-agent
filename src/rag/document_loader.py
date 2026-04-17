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
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
            chunk_size: max characters per chunk
            chunk_overlap: overlap between adjacent chunks (preserves context)
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self._loaded_files: set[str] = set()  # dedup tracker

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file and split into chunks. Returns [] on error."""
        path = Path(file_path)
        if not path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return []

        # Deduplication: skip if already indexed
        resolved = str(path.resolve())
        if resolved in self._loaded_files:
            console.print(f"[yellow]Already indexed, skipping: {path.name}[/yellow]")
            return []

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            console.print(
                f"[red]Unsupported format: {ext}. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}[/red]"
            )
            return []

        console.print(
            f"[cyan]Loading:[/cyan] {path.name} "
            f"[dim]({self.SUPPORTED_EXTENSIONS[ext]})[/dim]"
        )

        # TextLoader handles PDF, TXT, and MD (markdown is plain text)
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            # Both .txt and .md can be loaded as plain text
            loader = TextLoader(str(path), encoding="utf-8")

        raw_docs = loader.load()

        # Attach metadata
        for doc in raw_docs:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = ext

        chunks = self.text_splitter.split_documents(raw_docs)
        self._loaded_files.add(resolved)  # mark as indexed
        console.print(
            f"[green]Done:[/green] {len(chunks)} chunks from {path.name}"
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
            console.print(f"[yellow]️  目录中未找到支持的文件: {dir_path}[/yellow]")
            return []

        console.print(f"[blue] 扫描目录:[/blue] {dir_path} — 找到 {len(files)} 个文件")

        for f in files:
            try:
                docs = self.load_file(str(f))
                all_docs.extend(docs)
            except Exception as e:
                console.print(f"[red] 加载失败:[/red] {f.name} — {e}")

        console.print(
            f"\n[bold green] 目录加载完成:[/bold green] "
            f"共 {len(files)} 个文件，{len(all_docs)} 个文本块\n"
        )
        return all_docs
