from ddgs import DDGS

with DDGS() as ddgs:
    results = ddgs.text("Give detailed company overview of ValueAdd Research and Analytics Solutions LLP", max_results=5)
    print(results)
