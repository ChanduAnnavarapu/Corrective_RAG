from utils import decompose_to_sentences, refine_context
from state import RAGstate
from langchain_core.prompts import ChatPromptTemplate
from llm import get_llm

def Corrective_Retrieval(RAGstate)->RAGstate:
    question = RAGstate["Question"]
    refined_context = RAGstate["refined_context"]
    
    answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided refined bullets.\n"
            "If the bullets are empty or insufficient, say: 'I don't know based on the provided books.'",
        ),
        ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
    ]
    )
    
    answer_chain = answer_prompt | get_llm()
    answer = answer_chain.invoke({"question": question, "refined_context": refined_context})
    return {"answer": answer}
    
    