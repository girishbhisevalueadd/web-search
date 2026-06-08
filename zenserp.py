"""Zenserp web search via LangGraph ReAct agent (custom tool)."""
import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

QUERY = "give me list of peer companies of ValueAdd Research And Analytics LLP"


@tool
def zenserp_search(query: str) -> str:
    """Search Google via Zenserp SERP API and return top organic results."""
    r = requests.get(
        "https://app.zenserp.com/api/v2/search",
        headers={"apikey": os.getenv("ZENSERP_API_KEY")},
        params={"q": query, "num": 10, "search_engine": "google.com", "gl": "us", "hl": "en"},
        timeout=30,
    )
    data = r.json()
    results = data.get("organic", [])[:10]
    if not results:
        return f"No results / response: {str(data)[:500]}"
    return "\n---\n".join(
        f"{x.get('title','')}\n{x.get('url','')}\n{x.get('description','')}" for x in results
    )


llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(llm, [zenserp_search])

result = agent.invoke({"messages": [("user", QUERY)]})
print("=== ZENSERP ===")
print(result["messages"][-1].content)
