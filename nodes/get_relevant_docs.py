from state import RAGstate
from Rag_flow import load_vector_store

def get_relevant_docs(state:RAGstate):
    query = state["Question"]
    vector_store = load_vector_store()
    relevant_docs = vector_store.similarity_search(query, k=3)
    return { 
            "Docs": [doc.page_content for doc in relevant_docs if len(relevant_docs) > 0]
            }