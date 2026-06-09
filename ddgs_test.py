"""DuckDuckGo web search via LangGraph ReAct agent."""
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

QUERY = "Give me list of peer companies of ValueAdd Research And Analytics Solutions LLP"

wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", max_results=10, safesearch="off")
search = DuckDuckGoSearchResults(api_wrapper=wrapper, num_results=10, output_format="list")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(llm, [search])

result = agent.invoke({"messages": [("user", QUERY)]})
print("=== DUCKDUCKGO ===")
print(result["messages"][-1].content)
