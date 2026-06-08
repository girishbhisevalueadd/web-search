"""Exa neural web search via LangGraph ReAct agent."""
import os
from dotenv import load_dotenv
from exa_py import Exa
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

QUERY = "give me list of peer companies of ValueAdd Research And Analytics LLP"

exa = Exa(api_key=os.getenv("EXA_API_KEY"))


@tool
def exa_search(query: str) -> str:
    """Neural web search via Exa with content extraction."""
    res = exa.search(
        query=query,
        num_results=10,
        type="auto"
    )
    out = []
    for r in res.results:
        out.append(f"{r.title}\n{r.url}\n{(r.text or '')[:600]}")
    return "\n---\n".join(out) or "No results"


llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(llm, [exa_search])

result = agent.invoke({"messages": [("user", QUERY)]})
print("=== EXA ===")
print(result["messages"][-1].content)

