"""SerpStack web search via LangGraph ReAct agent (custom tool)."""
import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

QUERY = "give me list of peer companies of ValueAdd Research And Analytics LLP"


@tool
def serpstack_search(query: str) -> str:
    """Search Google via SerpStack and return top organic results."""
    r = requests.get(
        "https://api.serpstack.com/search",
        params={
            "access_key": os.getenv("SERPSTACK_API_KEY"),
            "query": query,
            "num": 10,
            "type": "web",
            "output": "json",
        },
        timeout=30,
    )
    data = r.json()
    if not data.get("success", True) and "error" in data:
        return f"SerpStack error: {data['error']}"
    results = data.get("organic_results", [])[:10]
    return "\n---\n".join(
        f"{x.get('title','')}\n{x.get('url','')}\n{x.get('snippet','')}" for x in results
    ) or "No results"


llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(llm, [serpstack_search])

result = agent.invoke({"messages": [("user", QUERY)]})
print("=== SERPSTACK ===")
print(result["messages"][-1].content)
