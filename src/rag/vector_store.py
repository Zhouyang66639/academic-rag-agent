"""
Vector Store Manager — FAISS + Document Store for Hybrid Retrieval

Embedding backend: FastEmbed (ONNX Runtime, no PyTorch dependency)
Persistence:
    vector_store/faiss/   -- FAISS index (binary)
    vector_store/docs.pkl -- raw Document list (for BM25 re-indexing)
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rich.console import Console

console = Console()


class VectorStoreManager:
    """
    Manages FAISS vector database + raw document persistence.

    Design decisions:
    - Uses FastEmbed (ONNX-based) instead of sentence-transformers to
      eliminate PyTorch as a dependency, reducing install size by ~1 GB.
    - Persists raw Document objects to disk so BM25 can be rebuilt on
      each session without re-processing source files.
    - Exposes get_hybrid_retriever() which returns a BM25+FAISS+RRF
      retriever when documents exist, else falls back to dense-only.
    """

    FAISS_DIR = "faiss"
    DOCS_FILE = "docs.pkl"

    def __init__(
        self,
        store_path: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.embeddings = self._init_embeddings(embedding_model)
        self.vector_store: Optional[FAISS] = None
        self._documents: List[Document] = []

        self._load()

    # ------------------------------------------------------------------
    # Embedding initialisation
    # ------------------------------------------------------------------

    def _init_embeddings(self, model_name: str):
        provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            console.print(f"[dim]Embedding: OpenAI ({model_name})[/dim]")
            return OpenAIEmbeddings(model=model_name)

        # Default: FastEmbed (ONNX, no PyTorch)
        try:
            from langchain_community.embeddings import FastEmbedEmbeddings
            console.print(
                f"[dim]Embedding: FastEmbed/{model_name} "
                f"(ONNX Runtime, no PyTorch)[/dim]"
            )
            return FastEmbedEmbeddings(model_name=model_name)
        except Exception:
            # Graceful fallback to HuggingFace if fastembed unavailable
            from langchain_community.embeddings import HuggingFaceEmbeddings
            console.print(
                f"[yellow]FastEmbed unavailable, falling back to "
                f"HuggingFace ({model_name})[/yellow]"
            )
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """Load FAISS index and raw documents from disk."""
        docs_path = self.store_path / self.DOCS_FILE
        if docs_path.exists():
            with open(docs_path, "rb") as f:
                self._documents = pickle.load(f)

        faiss_path = self.store_path / self.FAISS_DIR
        if faiss_path.exists() and any(faiss_path.iterdir()):
            try:
                self.vector_store = FAISS.load_local(
                    str(faiss_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                console.print(
                    f"[green]Vector store loaded:[/green] "
                    f"{self.vector_store.index.ntotal} vectors | "
                    f"{len(self._documents)} raw docs"
                )
            except Exception as e:
                console.print(f"[yellow]Failed to load store, rebuilding: {e}[/yellow]")
                self.vector_store = None

    def _save(self):
        """Persist FAISS index and raw documents."""
        faiss_path = self.store_path / self.FAISS_DIR
        faiss_path.mkdir(exist_ok=True)
        self.vector_store.save_local(str(faiss_path))

        docs_path = self.store_path / self.DOCS_FILE
        with open(docs_path, "wb") as f:
            pickle.dump(self._documents, f)

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            console.print("[yellow]No documents to add[/yellow]")
            return 0

        console.print(f"[cyan]Vectorizing {len(documents)} chunks...[/cyan]")

        # Keep raw docs for BM25
        self._documents.extend(documents)

        # Build / extend FAISS index
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        self._save()
        console.print(
            f"[green]Done.[/green] "
            f"Store: {self.vector_store.index.ntotal} vectors "
            f"({len(self._documents)} raw docs)"
        )
        return len(documents)

    def clear(self):
        import shutil
        if self.store_path.exists():
            shutil.rmtree(self.store_path)
        self.store_path.mkdir(parents=True)
        self.vector_store = None
        self._documents = []
        console.print("[yellow]Vector store cleared[/yellow]")

    # ------------------------------------------------------------------
    # Retriever factories
    # ------------------------------------------------------------------

    def get_dense_retriever(self, k: int = 4):
        """Dense-only FAISS MMR retriever."""
        if self.vector_store is None:
            return None
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3},
        )

    def get_hybrid_retriever(self, k: int = 4):
        """
        Hybrid BM25 + Dense retriever with RRF fusion.

        Falls back to dense-only if no documents are indexed.
        """
        if not self._documents:
            return None

        from src.rag.hybrid_retriever import HybridRetriever
        from langchain_community.retrievers import BM25Retriever

        bm25 = BM25Retriever.from_documents(self._documents, k=k)
        dense = self.get_dense_retriever(k=k)

        if dense is None:
            return bm25

        return HybridRetriever(
            bm25_retriever=bm25,
            dense_retriever=dense,
            k=k,
        )

    def get_retriever(self, k: int = 4):
        """Returns hybrid retriever if available, else dense-only."""
        return self.get_hybrid_retriever(k=k) or self.get_dense_retriever(k=k)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        if self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal
