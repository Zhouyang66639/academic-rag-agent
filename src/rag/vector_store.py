"""
Vector Store Module
基于 FAISS 的本地向量数据库，支持持久化存储和增量更新
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.progress import track

console = Console()


class VectorStoreManager:
    """
    FAISS 向量数据库管理器

    Features:
        - 本地持久化（无需运行外部服务）
        - 增量添加文档
        - 相似度检索 + MMR 多样性检索
    """

    def __init__(self, store_path: str, embedding_model: str = "text-embedding-3-small"):
        self.store_path = Path(store_path)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store: Optional[FAISS] = None
        self._load_existing()

    def _load_existing(self):
        """如果本地有保存的向量库，自动加载"""
        if self.store_path.exists() and any(self.store_path.iterdir()):
            try:
                self.vector_store = FAISS.load_local(
                    str(self.store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                doc_count = self.vector_store.index.ntotal
                console.print(
                    f"[green]✅ 已加载本地向量库:[/green] {doc_count} 个向量块 "
                    f"[dim]({self.store_path})[/dim]"
                )
            except Exception as e:
                console.print(f"[yellow]⚠️  加载向量库失败，将新建: {e}[/yellow]")
                self.vector_store = None

    def add_documents(self, documents: List[Document]) -> int:
        """
        向向量库中添加文档

        Returns:
            int: 新增的向量数量
        """
        if not documents:
            console.print("[yellow]⚠️  没有文档需要添加[/yellow]")
            return 0

        console.print(f"[cyan]🔄 正在向量化 {len(documents)} 个文本块...[/cyan]")

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        self._save()
        console.print(
            f"[green]✅ 向量化完成:[/green] "
            f"数据库现共有 [bold]{self.vector_store.index.ntotal}[/bold] 个向量块"
        )
        return len(documents)

    def similarity_search(
        self, query: str, k: int = 4, use_mmr: bool = True
    ) -> List[Document]:
        """
        语义搜索

        Args:
            query: 查询问题
            k: 返回结果数
            use_mmr: 使用 MMR 算法保证结果多样性（避免重复内容）
        """
        if self.vector_store is None:
            raise RuntimeError("向量库为空，请先加载文档！")

        if use_mmr:
            # MMR: 在相关性和多样性之间取平衡
            results = self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=k * 3
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)

        return results

    def get_retriever(self, k: int = 4):
        """返回 LangChain 兼容的检索器对象，方便接入 Chain"""
        if self.vector_store is None:
            raise RuntimeError("向量库为空，请先加载文档！")
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3},
        )

    def _save(self):
        """持久化保存到本地"""
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.store_path))

    def clear(self):
        """清空向量库"""
        self.vector_store = None
        if self.store_path.exists():
            import shutil
            shutil.rmtree(self.store_path)
        console.print("[yellow]🗑️  向量库已清空[/yellow]")

    @property
    def doc_count(self) -> int:
        """返回当前存储的向量数量"""
        if self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal
