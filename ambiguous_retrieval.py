from state import RAGstate
from web_search import web_search_node

def Ambiguous_Retrieval(state: RAGstate):
    web_search_results = web_search_node(state)
    
    return {
        **web_search_results,
    }