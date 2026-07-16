from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from llm import get_llm
from state import RAGstate
from langchain_core.prompts import ChatPromptTemplate
from web_search import web_search_node
from ambiguous_retrieval import Ambiguous_Retrieval
from generate_answer import generate_answer_node
from eval_doc_node import eval_doc_node
from typing import List,TypedDict
from Rag_flow import get_relevant_docs
from refine_context import decompose_to_sentences, refine_context_node
import re


def decide_corrective_action(state: RAGstate):
    if state["verdict"] == "INCORRECT":
        return "incorrect"
    elif state["verdict"] == "CORRECT":
        return "correct"
    else:
        return "ambiguous"
    

graph = StateGraph(RAGstate)

graph.add_node("get_relevant_docs", get_relevant_docs)
graph.add_node("refine_context", refine_context_node)
graph.add_node("eval_doc_node", eval_doc_node)
graph.add_node("web_search", web_search_node)
graph.add_node("ambiguous_retrieval", Ambiguous_Retrieval)
graph.add_node("generate_answer",generate_answer_node)

graph.add_edge(START, "get_relevant_docs")
graph.add_edge("get_relevant_docs", "eval_doc_node")
graph.add_conditional_edges("eval_doc_node", decide_corrective_action,{
    "correct": "refine_context",
    "incorrect": "web_search",
    "ambiguous": "ambiguous_retrieval"
})

graph.add_edge("ambiguous_retrieval", "refine_context")
graph.add_edge("web_search", "refine_context")
graph.add_edge("refine_context", "generate_answer")
graph.add_edge("generate_answer", END)

workflow = graph.compile()

initial_state = RAGstate(
    Question="What is ml?",
    Docs=[],
    strips=[],
    kept_strips=[],
    web_docs=[],
    refined_context="",
    answer=""
)

agent=workflow.invoke(initial_state)
print("Final State:", agent)


