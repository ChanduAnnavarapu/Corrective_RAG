from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults
from state import RAGstate
from llm import llm_invoke
from dotenv import load_dotenv
import os

load_dotenv()

tavily = TavilySearchResults(max_results=5,
                             tavily_api_key=os.getenv("TAVILY_API_KEY"))

def web_search_node(state: RAGstate) -> RAGstate:

    q = state["Question"]  # no query rewrite
    results = tavily.invoke({"query": q})  # no knowledge selection

    web_docs = []
    for r in results or []:

        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}
