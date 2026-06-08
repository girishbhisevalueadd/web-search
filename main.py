from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------------------
# SEARCH
# ---------------------------

tavily = TavilyClient()

search_result = tavily.search(
    query="ValueAdd Research and Analytics Solutions LLP company overview",
    max_results=5
)

context = ""

for result in search_result["results"]:
    context += f"""
    Title: {result['title']}
    Content: {result['content']}
    URL: {result['url']}
    """

# ---------------------------
# PROMPT
# ---------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a professional research analyst.

     Rules:
     - Use ONLY provided context
     - Do NOT invent information
     - If information is unavailable, say so
     - Give structured output
     """),

    ("human",
     """
     Context:
     {context}

     Question:
     {question}
     """)
])

# ---------------------------
# MODEL
# ---------------------------

model = ChatAnthropic(
    model="claude-opus-4-7",
    max_tokens=3000
)

chain = prompt | model

response = chain.invoke({
    "context": context,
    "question": "Give detailed company overview"
})

print(response.content)