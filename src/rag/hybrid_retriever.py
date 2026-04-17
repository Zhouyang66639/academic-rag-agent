"""
Hybrid Retriever — BM25 (sparse) + FAISS (dense) with RRF Fusion

Reciprocal Rank Fusion formula (Cormack, Clarke & Buettcher, SIGIR 2009):
    score(d) = sum_i [ 1 / (k + rank_i(d)) ]

where k=60 is the standard smoothing constant. Documents ranked highly by
either retriever get high final scores; documents in both get a bonus.
"""

from typing import List, Optional, Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field


class HybridRetriever(BaseRetriever):
    """
    Hybrid sparse + dense retriever with Reciprocal Rank Fusion (RRF).

    Pipeline:
        query --> BM25 (keyword match) --> ranked list A
              --> FAISS MMR (semantic)  --> ranked list B
                                              |
                                        RRF fusion
                                              |
                                        top-k results

    Why hybrid?
    - BM25 excels at exact keyword / named entity matching (model names, authors)
    - Dense vectors excel at semantic / paraphrase matching
    - RRF balances both without requiring score calibration

    Attributes:
        bm25_retriever: sparse BM25 retriever
        dense_retriever: dense FAISS retriever
        k: number of final documents to return
        rrf_k: RRF smoothing constant (default 60, per original paper)
    """

    bm25_retriever: Any = Field(description="BM25 sparse retriever")
    dense_retriever: Any = Field(description="Dense vector retriever")
    k: int = Field(default=4, description="Final top-k results")
    rrf_k: int = Field(default=60, description="RRF smoothing constant")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        # Retrieve from both sources independently
        bm25_docs = self.bm25_retriever.invoke(query)
        dense_docs = self.dense_retriever.invoke(query)
        return self._rrf_fusion(bm25_docs, dense_docs)

    def _rrf_fusion(
        self,
        bm25_docs: List[Document],
        dense_docs: List[Document],
    ) -> List[Document]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        Each document's score accumulates 1/(rrf_k + rank) for every list
        it appears in. This naturally handles documents absent from one list
        without requiring score normalization.
        """
        def fingerprint(doc: Document) -> str:
            """Stable document identity based on content prefix."""
            return doc.page_content[:200]

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            fid = fingerprint(doc)
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[fid] = doc

        for rank, doc in enumerate(dense_docs):
            fid = fingerprint(doc)
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[fid] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[fid] for fid, _ in ranked[: self.k]]
