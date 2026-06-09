"""
Claude.ai-research-mode quality web research agent.

Architecture (LangGraph state machine):
    Plan -> Search -> Read -> Reflect -> (loop to Search | Synthesize) -> END

- Search fallback chain: SerpStack -> Zenserp -> Serper-Dev -> Tavily -> DuckDuckGo
- Exa runs in parallel with the SERP chain every time (neural complement).
- Page reader: Firecrawl primary, Tavily Extract fallback.
- Reasoning LLM: Claude Opus 4.8 for Plan / Reflect / Synthesize.
"""
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypedDict

warnings.filterwarnings("ignore", category=UserWarning)

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from exa_py import Exa
from firecrawl import FirecrawlApp
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchResults  # noqa: F401
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    GoogleSerperAPIWrapper,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilyExtract, TavilySearch
from langgraph.graph import END, START, StateGraph

# ============================================================
# Config
# ============================================================
MAX_ITERATIONS = 4
PAGES_PER_ITER = 8
PAGE_CHAR_BUDGET = 12_000  # ~3k tokens per page

LLM = ChatAnthropic(model="claude-opus-4-8", max_tokens=8192)
exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))
firecrawl_client = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


# ============================================================
# Graph state
# ============================================================
class ResearchState(TypedDict):
    query: str
    sub_queries: list[str]
    search_results: list[dict]   # accumulated, deduped by URL
    fetched_pages: list[dict]    # {url, content}
    iteration: int
    final_answer: str


# ============================================================
# Structured-output schemas (Claude returns these as JSON)
# ============================================================
class Plan(BaseModel):
    query_class: Literal[
        "entity_research", "factual", "current_events",
        "comparison", "how_to", "general",
    ]
    sub_queries: list[str] = Field(min_length=2, max_length=5)


class Reflection(BaseModel):
    needs_more: bool
    new_sub_queries: list[str] = Field(default_factory=list)
    reasoning: str


# ============================================================
# Search providers — each returns list[{title, url, snippet}]
# ============================================================
def _serpstack(q: str) -> list[dict]:
    try:
        r = requests.get(
            "https://api.serpstack.com/search",
            params={
                "access_key": os.getenv("SERPSTACK_API_KEY"),
                "query": q, "num": 10, "type": "web",
            },
            timeout=20,
        )
        d = r.json()
        return [
            {"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": x.get("snippet", "")}
            for x in (d.get("organic_results") or [])
        ][:10]
    except Exception:
        return []


def _zenserp(q: str) -> list[dict]:
    try:
        r = requests.get(
            "https://app.zenserp.com/api/v2/search",
            headers={"apikey": os.getenv("ZENSERP_API_KEY")},
            params={"q": q, "num": 10, "search_engine": "google.com"},
            timeout=20,
        )
        d = r.json()
        return [
            {"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": x.get("description", "")}
            for x in (d.get("organic") or [])
        ][:10]
    except Exception:
        return []


def _serper_dev(q: str) -> list[dict]:
    try:
        w = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_DEV_API_KEY"), k=10)
        data = w.results(q)
        return [
            {"title": x.get("title", ""), "url": x.get("link", ""),
             "snippet": x.get("snippet", "")}
            for x in data.get("organic", [])
        ][:10]
    except Exception:
        return []


def _tavily(q: str) -> list[dict]:
    try:
        t = TavilySearch(max_results=10, search_depth="advanced")
        data = t.invoke({"query": q})
        return [
            {"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": x.get("content", "")}
            for x in data.get("results", [])
        ][:10]
    except Exception:
        return []


def _ddg(q: str) -> list[dict]:
    try:
        w = DuckDuckGoSearchAPIWrapper(max_results=10)
        results = w.results(q, 10)
        return [
            {"title": x.get("title", ""), "url": x.get("link", ""),
             "snippet": x.get("snippet", "")}
            for x in results
        ][:10]
    except Exception:
        return []


def _exa(q: str) -> list[dict]:
    """Neural search — runs in parallel with the SERP chain, not part of fallback."""
    try:
        res = exa_client.search_and_contents(
            query=q, num_results=10, type="auto",
            text={"max_characters": 600},
        )
        return [
            {"title": r.title or "", "url": r.url or "",
             "snippet": (r.text or "")[:500]}
            for r in res.results
        ][:10]
    except Exception:
        return []


# Fallback order per user spec: SerpStack → Zenserp → Serper-Dev → Tavily → DDG
SERP_CHAIN = [_serpstack, _zenserp, _serper_dev, _tavily, _ddg]


def serp_with_fallback(q: str) -> list[dict]:
    """Try providers in order; return first non-empty result set."""
    for fn in SERP_CHAIN:
        results = fn(q)
        if results:
            return results
    return []


# ============================================================
# Page reader — Firecrawl primary, Tavily Extract fallback
# ============================================================
def fetch_page(url: str) -> dict:
    try:
        res = firecrawl_client.scrape(url, formats=["markdown"])
        content = (
            getattr(res, "markdown", None)
            or (res.get("markdown") if isinstance(res, dict) else None)
            or ""
        )
        if content:
            return {"url": url, "content": content[:PAGE_CHAR_BUDGET]}
    except Exception:
        pass
    try:
        ext = TavilyExtract()
        data = ext.invoke({"urls": [url]})
        results = data.get("results") if isinstance(data, dict) else None
        if results:
            content = results[0].get("raw_content") or results[0].get("content") or ""
            if content:
                return {"url": url, "content": content[:PAGE_CHAR_BUDGET]}
    except Exception:
        pass
    return {"url": url, "content": ""}


# ============================================================
# Nodes
# ============================================================
def plan_node(state: ResearchState) -> dict:
    planner = LLM.with_structured_output(Plan)
    out: Plan = planner.invoke([
        SystemMessage(content=(
            "You are a research planner. Decompose the user's question into 2-5 "
            "atomic web-search sub-queries that together will gather enough "
            "evidence to answer it thoroughly and accurately. First understand "
            "WHAT the entity/topic actually is before searching for details "
            "about it. Classify the query type."
        )),
        HumanMessage(content=state["query"]),
    ])
    print(f"[PLAN]    class={out.query_class}")
    for q in out.sub_queries:
        print(f"          - {q}")
    return {"sub_queries": out.sub_queries, "iteration": 1}


def search_node(state: ResearchState) -> dict:
    queries = state["sub_queries"]
    prior = list(state.get("search_results") or [])
    seen = {r["url"] for r in prior}
    new_results = []

    with ThreadPoolExecutor(max_workers=10) as ex:
        serp_futures = [ex.submit(serp_with_fallback, q) for q in queries]
        exa_futures = [ex.submit(_exa, q) for q in queries]
        for fut in serp_futures + exa_futures:
            for r in fut.result():
                if r["url"] and r["url"] not in seen:
                    seen.add(r["url"])
                    new_results.append(r)

    print(f"[SEARCH]  +{len(new_results)} new results (total {len(prior) + len(new_results)})")
    return {"search_results": prior + new_results}


def read_node(state: ResearchState) -> dict:
    fetched_urls = {p["url"] for p in (state.get("fetched_pages") or [])}
    candidates = [r for r in state["search_results"] if r["url"] not in fetched_urls]
    to_fetch = candidates[:PAGES_PER_ITER]

    new_pages = []
    with ThreadPoolExecutor(max_workers=PAGES_PER_ITER) as ex:
        for page in ex.map(fetch_page, [r["url"] for r in to_fetch]):
            if page["content"]:
                new_pages.append(page)

    print(f"[READ]    +{len(new_pages)} pages (total {len(fetched_urls) + len(new_pages)})")
    return {"fetched_pages": list(state.get("fetched_pages") or []) + new_pages}


def reflect_node(state: ResearchState) -> dict:
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"[REFLECT] iteration cap ({MAX_ITERATIONS}) reached — synthesizing")
        return {"sub_queries": []}

    critic = LLM.with_structured_output(Reflection)
    # Only show the freshest pages to the critic to keep context tight.
    recent = state["fetched_pages"][-PAGES_PER_ITER:]
    context = "\n\n---\n\n".join(
        f"URL: {p['url']}\n{p['content'][:2500]}" for p in recent
    )
    out: Reflection = critic.invoke([
        SystemMessage(content=(
            "You are a research critic. Given the original question and evidence "
            "gathered so far, decide whether the evidence is sufficient to write "
            "a thorough, accurate answer. If gaps exist, propose 1-3 NEW "
            "sub-queries that fill them. Be strict: if information is shallow, "
            "unverified, or one-sided, say needs_more=true."
        )),
        HumanMessage(content=(
            f"Original question: {state['query']}\n\n"
            f"Iteration: {state['iteration']} / {MAX_ITERATIONS}\n\n"
            f"Evidence so far:\n{context}"
        )),
    ])
    print(f"[REFLECT] needs_more={out.needs_more}")
    if out.needs_more and out.new_sub_queries:
        for q in out.new_sub_queries:
            print(f"          + {q}")
        return {
            "sub_queries": out.new_sub_queries,
            "iteration": state["iteration"] + 1,
        }
    return {"sub_queries": []}


def synthesize_node(state: ResearchState) -> dict:
    context = "\n\n---\n\n".join(
        f"[Source: {p['url']}]\n{p['content']}" for p in state["fetched_pages"]
    )
    answer = LLM.invoke([
        SystemMessage(content=(
            "You are a research analyst writing a thorough, accurate, "
            "well-grounded answer. Synthesize ONLY from the evidence below — "
            "do not invent facts. Where a specific claim depends on a specific "
            "source, mention the source URL inline. End with a numbered list of "
            "all sources used."
        )),
        HumanMessage(content=f"Question: {state['query']}\n\nEvidence:\n{context}"),
    ])
    text = answer.content if isinstance(answer.content, str) else str(answer.content)
    return {"final_answer": text}


# ============================================================
# Build graph
# ============================================================
def _route(state: ResearchState) -> str:
    return "search" if state["sub_queries"] else "synthesize"


def build_agent():
    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("search", search_node)
    g.add_node("read", read_node)
    g.add_node("reflect", reflect_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "read")
    g.add_edge("read", "reflect")
    g.add_conditional_edges("reflect", _route, {"search": "search", "synthesize": "synthesize"})
    g.add_edge("synthesize", END)
    return g.compile()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import sys

    QUERY = " ".join(sys.argv[1:]) or (
        "Give me list of peer companies of ValueAdd Research And Analytics Solutions LLP"
    )

    agent = build_agent()
    print(f"\n[QUERY]   {QUERY}\n")
    result = agent.invoke(
        {
            "query": QUERY,
            "sub_queries": [],
            "search_results": [],
            "fetched_pages": [],
            "iteration": 0,
            "final_answer": "",
        },
        config={"recursion_limit": 50},
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60 + "\n")
    print(result["final_answer"])

    with open("research_output.txt", "w", encoding="utf-8") as f:
        f.write(f"QUERY: {QUERY}\n\n{result['final_answer']}\n")
    print("\n[saved to research_output.txt]")
