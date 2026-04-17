"""
arXiv Search Tool
给 Agent 提供实时搜索 arXiv 论文的能力
"""

import arxiv
from typing import List, Dict
from langchain.tools import tool
from rich.console import Console

console = Console()


def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    搜索 arXiv 论文

    Args:
        query: 搜索关键词（支持英文，如 'RAG large language model'）
        max_results: 返回结果数量

    Returns:
        包含标题、摘要、链接、作者的论文列表
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "authors": [a.name for a in paper.authors[:3]],  # 最多显示3位作者
            "abstract": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
            "url": paper.entry_id,
            "published": paper.published.strftime("%Y-%m-%d"),
            "categories": paper.categories[:3],
        })

    return results


def format_arxiv_results(results: List[Dict]) -> str:
    """将搜索结果格式化为可读文本"""
    if not results:
        return "未找到相关论文。"

    lines = [f"找到 {len(results)} 篇相关论文：\n"]
    for i, paper in enumerate(results, 1):
        authors_str = ", ".join(paper["authors"])
        if len(paper["authors"]) >= 3:
            authors_str += " et al."
        lines.append(
            f"**[{i}] {paper['title']}**\n"
            f"   👤 作者: {authors_str}\n"
            f"   📅 发布: {paper['published']}\n"
            f"   🏷️  领域: {', '.join(paper['categories'])}\n"
            f"   📝 摘要: {paper['abstract']}\n"
            f"   🔗 链接: {paper['url']}\n"
        )
    return "\n".join(lines)


# LangChain Tool 包装，方便 Agent 直接调用
@tool
def arxiv_search_tool(query: str) -> str:
    """
    搜索 arXiv 学术论文数据库。
    当用户询问某个研究领域的最新论文、想了解某个概念的相关工作时使用此工具。
    输入应为英文关键词，例如: 'RAG retrieval augmented generation survey'

    Args:
        query: 英文搜索关键词

    Returns:
        格式化的论文搜索结果
    """
    console.print(f"[dim]🔍 搜索 arXiv: {query}[/dim]")
    try:
        results = search_arxiv(query, max_results=5)
        return format_arxiv_results(results)
    except Exception as e:
        return f"arXiv 搜索失败: {e}。请检查网络连接。"
