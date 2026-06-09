"""Serper-Dev (alternate Serper key) web search via LangGraph ReAct agent."""
import os
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

QUERY = "Give me list of peer companies of ValueAdd Research And Analytics Solutions LLP"

serper = GoogleSerperAPIWrapper(
    serper_api_key=os.getenv("SERPER_DEV_API_KEY"),
    k=10,
    type="search",
    gl="us",
    hl="en",
)
tool = Tool(
    name="google_serper_dev_search",
    description="Google search via Serper (dev key). Use for current/real-time info.",
    func=serper.run,
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(llm, [tool])

result = agent.invoke({"messages": [("user", QUERY)]})
print("=== SERPER-DEV ===")
print(result["messages"][-1].content)
