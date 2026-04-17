"""
Bulk arXiv Indexer — Fetch and index hundreds of paper abstracts at scale

Strategy:
    Instead of downloading full PDFs, we fetch paper metadata + abstract via
    the arXiv API and create Document objects for indexing. This allows
    ingesting 100+ papers in seconds with minimal storage overhead.

    Each document contains:
        Title / Authors / Published date / Categories / Abstract / URL
"""

import arxiv
from typing import List
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()


def bulk_fetch_arxiv(query: str, max_results: int = 100) -> List[Document]:
    """
    Fetch up to max_results papers from arXiv and return as Documents.

    Uses abstract-only strategy (no PDF download) for speed and storage efficiency.
    Each Document is structured for BM25 + FAISS hybrid indexing.

    Args:
        query: arXiv search query (English)
        max_results: number of papers to fetch (max ~2000 via arXiv API)

    Returns:
        List of Document objects ready for vector store indexing
    """
    client = arxiv.Client(
        page_size=50,           # fetch 50 at a time from arXiv API
        delay_seconds=1.0,      # respect arXiv rate limits
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    documents: List[Document] = []

    console.print(
        f"[cyan]Fetching up to {max_results} papers from arXiv:[/cyan] {query}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading metadata...", total=max_results)

        for paper in client.results(search):
            # Format content for BM25 + semantic search
            authors = ", ".join(a.name for a in paper.authors[:5])
            if len(paper.authors) > 5:
                authors += " et al."

            content = (
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"Categories: {', '.join(paper.categories[:3])}\n"
                f"Abstract: {paper.summary}\n"
                f"URL: {paper.entry_id}"
            )

            doc = Document(
                page_content=content,
                metadata={
                    "source": paper.entry_id,
                    "title": paper.title,
                    "authors": authors,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "source_file": f"arxiv:{paper.get_short_id()}",
                    "file_type": ".arxiv",
                },
            )
            documents.append(doc)
            progress.advance(task)

            if len(documents) >= max_results:
                break

    console.print(
        f"[green]Fetched {len(documents)} papers from arXiv[/green]"
    )
    return documents
