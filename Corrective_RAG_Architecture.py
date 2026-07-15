from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from llm import get_llm
from state import RAGstate
from state import keepOrDrop
from langchain_core.prompts import ChatPromptTemplate
from correct_retrieval import Corrective_Retrieval
from incorrect_retrieval import Incorrect_Retrieval
from ambiguous_retrieval import Ambiguous_Retrieval
from eval_doc_node import eval_doc_node
from typing import List,TypedDict
from Rag_flow import get_relevant_docs
from utils import decompose_to_sentences, refine_context
import re


def decide_corrective_action(state: RAGstate):
    if state["verdict"] == "INCORRECT":
        return "incorrect"
    elif state["verdict"] == "CORRECT":
        return "corrective"
    else:
        return "ambiguous"
    

graph = StateGraph(RAGstate)

graph.add_node("get_relevant_docs", get_relevant_docs)
graph.add_node("refine_context", refine_context)
graph.add_node("eval_doc_node", eval_doc_node)
graph.add_node("corrective_retrieval", Corrective_Retrieval)
graph.add_node("incorrect_retrieval", Incorrect_Retrieval)
graph.add_node("ambiguous_retrieval", Ambiguous_Retrieval)

graph.add_edge(START, "get_relevant_docs")
graph.add_edge("get_relevant_docs", "refine_context")
graph.add_edge("refine_context", "eval_doc_node")
graph.add_conditional_edges("eval_doc_node", decide_corrective_action,{
    "corrective": "corrective_retrieval",
    "incorrect": "incorrect_retrieval",
    "ambiguous": "ambiguous_retrieval"
})
graph.add_edge("corrective_retrieval", END)
graph.add_edge("incorrect_retrieval", END)
graph.add_edge("ambiguous_retrieval", END)

workflow = graph.compile()

initial_state = RAGstate(
    Question="What is ml?",
    Docs=[],
    strips=[],
    kept_strips=[],
    refined_context="",
    answer=""
)

agent=workflow.invoke(initial_state)
print("Final State:", agent)


