"""Anthropic Claude with native web_search tool via LangChain ChatAnthropic.

NOTE: file name `anthropic.py` SHADOWS the installed `anthropic` package and
will break `from langchain_anthropic import ChatAnthropic`. Rename to
`anthropic_test.py` or run from a different working directory.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

QUERY = "give me list of peer companies of ValueAdd Research And Analytics LLP"

# Claude Opus 4.8 is the latest, most capable model. web_search_20250305 is
# Anthropic's server-side web search tool.
llm = ChatAnthropic(
    model="claude-opus-4-8",
    max_tokens=4096,
).bind_tools([{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}])

response = llm.invoke(QUERY)
print("=== ANTHROPIC (native web_search) ===")
print(response.content if isinstance(response.content, str) else response.text())
