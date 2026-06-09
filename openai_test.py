from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

QUERY = "Give me list of peer companies of ValueAdd Research And Analytics Solutions LLP"

# gpt-4o supports the built-in web_search tool via the Responses API.
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    use_responses_api=True,
    output_version="responses/v1",
).bind_tools([{"type": "web_search_preview"}])

response = llm.invoke(QUERY)
print("=== OPENAI (native web_search) ===")
print(response.content if isinstance(response.content, str) else response.text())


