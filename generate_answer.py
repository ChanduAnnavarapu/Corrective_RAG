from state import RAGstate
from llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

def generate_answer_node(state: RAGstate):
    llm = get_llm()
    prompt_template = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context: {context}\n\n"
        "Question: {Question}\n\n"
        "Answer:"
    )
    prompt = prompt_template.invoke({"context": state['refined_context'], "Question": state['Question']})
    answer = get_llm().invoke(prompt)
    return {'answer':answer.content}