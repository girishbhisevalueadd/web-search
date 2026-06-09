"""Firecrawl web search via LangGraph ReAct agent."""
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._fields")
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

QUERY = "Give me list of peer companies of ValueAdd Research And Analytics Solutions LLP"

fc = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


@tool
def firecrawl_search(query: str) -> str:
    """Search the web with Firecrawl and return top results with scraped content."""
    res = fc.search(query=query, limit=10, scrape_options={"formats": ["markdown"]})
    items = res.get("data", []) if isinstance(res, dict) else getattr(res, "data", [])
    out = []
    for r in items[:10]:
        d = r if isinstance(r, dict) else r.__dict__
        out.append(f"{d.get('title','')}\n{d.get('url','')}\n{(d.get('markdown') or d.get('description',''))[:600]}")
    return "\n---\n".join(out) or "No results"


llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(llm, [firecrawl_search])

result = agent.invoke({"messages": [("user", QUERY)]})
print("=== FIRECRAWL ===")
print(result["messages"][-1].content)
