"""Tavily web search via LangGraph ReAct agent."""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

load_dotenv()

QUERY = "give me list of peer companies of ValueAdd Research And Analytics LLP"

search = TavilySearch(
    max_results=10,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(llm, [search])

result = agent.invoke({"messages": [("user", QUERY)]})
print("=== TAVILY ===")
print(result["messages"][-1].content)
