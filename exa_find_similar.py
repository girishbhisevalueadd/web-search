"""Find peer companies via Exa's find_similar (semantic neighbor URLs)."""
import os
from dotenv import load_dotenv
from exa_py import Exa
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# Anchor URL — the company we want peers of.
TARGET_URL = "https://www.valueaddresearch.com/"

exa = Exa(api_key=os.getenv("EXA_API_KEY"))

# Pull semantically similar pages (Exa's neural index, not keyword match).
# exclude_source_domain=True drops valueaddresearch.com itself from results.
res = exa.find_similar_and_contents(
    url=TARGET_URL,
    num_results=15,
    exclude_source_domain=True,
    text={"max_characters": 800},
    highlights={"num_sentences": 3, "highlights_per_url": 2},
    category="company",
)

# Build a compact context block for the LLM to synthesize.
context = "\n\n---\n\n".join(
    f"URL: {r.url}\nTITLE: {r.title}\nSNIPPET: {(r.text or '')[:600]}"
    for r in res.results
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
response = llm.invoke([
    SystemMessage(content=(
        "You are a research analyst. Given a set of company web pages that are "
        "semantically similar to a target company, produce a clean list of true "
        "PEER COMPANIES — same industry, similar service offering, comparable scale. "
        "Drop universities, news sites, directories, clients, and unrelated firms. "
        "For each peer give: name, one-line description, URL."
    )),
    HumanMessage(content=(
        f"Target company: {TARGET_URL} (ValueAdd Research And Analytics — "
        f"investment research / KPO firm based in India).\n\n"
        f"Candidate peer pages from Exa find_similar:\n\n{context}"
    )),
])

print("=== EXA find_similar peers of ValueAdd ===\n")
print(response.content)
