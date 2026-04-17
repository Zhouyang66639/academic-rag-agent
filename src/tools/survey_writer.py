"""
Survey Writer — RAG-based Academic Survey Generation Pipeline

Architecture (Map-Reduce):

    Phase 1 — INDEX
        Bulk arXiv fetch (100+ papers) -> vector store

    Phase 2 — MAP (per section)
        Section topic -> Hybrid Retriever -> top-k relevant chunks
                      -> LLM -> section draft

    Phase 3 — REDUCE
        All section drafts -> assembled Markdown document -> saved to file

This approach works within LLM context limits while synthesising
information from hundreds of papers stored in the vector index.

Output format: structured academic survey in Markdown
    1. Abstract
    2. Introduction
    3. Background & Preliminaries
    4. Taxonomy of Approaches
    5. Core Methods & Techniques
    6. Applications
    7. Open Challenges & Future Directions
    8. Conclusion
    9. References (papers used by RAG retrieval)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

console = Console()

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

SURVEY_SYSTEM = """You are an expert academic writer specializing in comprehensive AI literature surveys.

Writing style:
- Clear, precise, academic English
- Objective tone; avoid marketing language
- Compare and contrast approaches critically
- Cite papers as [Author et al., YEAR] where information comes from retrieved papers
- Each section should be self-contained but coherent with the whole

You will receive snippets from relevant papers to ground your writing."""

SECTION_PROMPTS = [
    (
        "abstract",
        "Write a concise academic Abstract (≈150 words) for a survey titled "
        "'A Survey on {topic}'. Cover: scope, methodology, key findings, and significance.",
        100,  # fetch_k for retrieval
    ),
    (
        "introduction",
        "Write the Introduction section (≈500 words) for a survey on '{topic}'. "
        "Cover: motivation, why this topic is important now, scope and structure of the survey.",
        120,
    ),
    (
        "background",
        "Write a Background & Preliminaries section (≈500 words) for a survey on '{topic}'. "
        "Define core concepts, terminology, and foundational methods that readers need.",
        120,
    ),
    (
        "taxonomy",
        "Write a Taxonomy of Approaches section (≈600 words) for a survey on '{topic}'. "
        "Organize existing work into clear categories with rationale for the classification.",
        150,
    ),
    (
        "methods",
        "Write a Core Methods & Techniques section (≈700 words) for a survey on '{topic}'. "
        "Describe major methodological families, compare their trade-offs, and discuss key results.",
        200,
    ),
    (
        "applications",
        "Write an Applications section (≈400 words) for a survey on '{topic}'. "
        "Describe real-world use cases and domains where these methods are deployed.",
        100,
    ),
    (
        "challenges",
        "Write an Open Challenges & Future Directions section (≈500 words) for a survey on '{topic}'. "
        "Identify unsolved problems, current limitations, and promising research directions.",
        120,
    ),
    (
        "conclusion",
        "Write a Conclusion section (≈200 words) for a survey on '{topic}'. "
        "Summarise key findings and provide a forward-looking closing statement.",
        80,
    ),
]


# ------------------------------------------------------------------
# Core pipeline
# ------------------------------------------------------------------

def generate_survey(
    topic: str,
    retriever,
    llm: "ChatOpenAI",
    output_path: Optional[str] = None,
    top_k: int = 5,
) -> tuple[str, str]:
    """
    Generate a full academic survey on a topic using RAG.

    Args:
        topic:       Research topic (e.g. "Retrieval-Augmented Generation for LLMs")
        retriever:   Hybrid BM25+FAISS retriever from VectorStoreManager
        llm:         ChatOpenAI instance to use for generation
        output_path: Where to save the Markdown file (auto-named if None)
        top_k:       Number of document chunks to retrieve per section

    Returns:
        (survey_text, output_filepath)
    """
    console.print(
        Panel(
            f"[bold cyan]Generating survey: {topic}[/bold cyan]\n"
            f"Sections: {len(SECTION_PROMPTS)} | Top-K per section: {top_k}",
            border_style="cyan",
        )
    )

    written_sections: dict[str, str] = {}
    all_sources: set[str] = set()  # collect cited papers

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description:<30}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Writing sections...", total=len(SECTION_PROMPTS)
        )

        for section_id, prompt_template, _ in SECTION_PROMPTS:
            progress.update(task, description=f"Writing: {section_id:<20}")

            # Build retrieval query for this section
            section_query = f"{topic} {section_id.replace('_', ' ')}"
            docs = retriever.invoke(section_query)[:top_k]

            # Collect source papers
            for doc in docs:
                src = doc.metadata.get("title") or doc.metadata.get("source_file", "")
                if src:
                    all_sources.add(src)

            # Build context from retrieved chunks
            context_parts = []
            for i, doc in enumerate(docs, 1):
                title = doc.metadata.get("title", f"Paper {i}")
                snippet = doc.page_content[:600]
                context_parts.append(f"[Paper {i}: {title}]\n{snippet}")
            context = "\n\n---\n\n".join(context_parts)

            # Generate section
            prompt = prompt_template.format(topic=topic)
            full_prompt = (
                f"## Retrieved papers for context:\n\n{context}\n\n"
                f"---\n\n## Task:\n{prompt}"
            )

            try:
                response = llm.invoke(
                    [
                        SystemMessage(content=SURVEY_SYSTEM),
                        HumanMessage(content=full_prompt),
                    ]
                )
                written_sections[section_id] = response.content
            except Exception as e:
                written_sections[section_id] = f"*[Generation error: {e}]*"

            progress.advance(task)

    # Assemble into final document
    survey_text = _assemble(topic, written_sections, all_sources)

    # Save to file
    if output_path is None:
        safe = topic.replace(" ", "_")[:40]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = f"survey_{safe}_{timestamp}.md"

    Path(output_path).write_text(survey_text, encoding="utf-8")
    console.print(
        Panel(
            f"[bold green]Survey complete![/bold green]\n"
            f"Saved to: [cyan]{output_path}[/cyan]\n"
            f"Sources referenced: {len(all_sources)} papers",
            border_style="green",
        )
    )

    return survey_text, output_path


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def _assemble(topic: str, sections: dict[str, str], sources: set[str]) -> str:
    date = datetime.now().strftime("%B %Y")

    references = ""
    if sources:
        references = "\n## References\n\n"
        for i, src in enumerate(sorted(sources), 1):
            references += f"[{i}] {src}\n"
        references += (
            "\n*Note: Full bibliographic details available in the indexed papers. "
            "Verify citations before academic submission.*\n"
        )

    return f"""# A Survey on {topic}

*Automatically generated by Academic RAG Agent · {date}*
*Based on RAG synthesis from indexed papers — verify before academic use.*

---

## Abstract

{sections.get("abstract", "")}

---

## 1. Introduction

{sections.get("introduction", "")}

---

## 2. Background and Preliminaries

{sections.get("background", "")}

---

## 3. Taxonomy of Approaches

{sections.get("taxonomy", "")}

---

## 4. Core Methods and Techniques

{sections.get("methods", "")}

---

## 5. Applications

{sections.get("applications", "")}

---

## 6. Open Challenges and Future Directions

{sections.get("challenges", "")}

---

## 7. Conclusion

{sections.get("conclusion", "")}

---
{references}
"""
