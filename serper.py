"""Serper (google.serper.dev) web search via LangGraph ReAct agent."""
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

QUERY = "give me list of peer companies of ValueAdd Research And Analytics LLP"

serper = GoogleSerperAPIWrapper(k=10, type="search", gl="us", hl="en")
tool = Tool(
    name="google_serper_search",
    description="Google search via Serper. Use for current/real-time info.",
    func=serper.run,
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(llm, [tool])

result = agent.invoke({"messages": [("user", QUERY)]})
print("=== SERPER ===")
print(result["messages"][-1].content)
