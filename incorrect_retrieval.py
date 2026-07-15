from state import RAGstate
from llm import llm_invoke

def Incorrect_Retrieval(state:RAGstate):
    ans=llm_invoke(state["Question"])
    return {"answer": ans}

