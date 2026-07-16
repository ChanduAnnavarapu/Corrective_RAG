from typing import List
import re
from xml.dom.minidom import Document
from state import RAGstate
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm import get_llm
from dotenv import load_dotenv
import os
load_dotenv()

class keepOrDrop(BaseModel):
    keep: bool=Field(description="True if the sentence is relevant to the question, False otherwise.")

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def refine_context_node(state: RAGstate)->RAGstate:
    good_docs = state["good_docs"]
    web_docs_with_info = state["web_docs"]
    web_docs = [doc.page_content for doc in web_docs_with_info]
    
    if state["verdict"] == "CORRECT":
        context = "\n\n".join(good_docs).strip()
    elif state["verdict"] == "INCORRECT":
        context = "\n\n".join(web_docs).strip()
    else:
        context = "\n\n".join(good_docs + web_docs).strip()

    filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict relevance filter.Determine whether the sentence is relevant to the question.

            Reply with exactly one word:
            true
            or
            false

            Do not provide any explanation.""",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
    )

    filter_chain = filter_prompt | get_llm().with_structured_output(keepOrDrop)
    
    strips = decompose_to_sentences(context)
    kept_strips = []
    
    if len(strips) > 0:
        for strip in strips:
            if filter_chain.invoke({"question": state["Question"], "sentence": strip}).keep:
                kept_strips.append(strip)
            
    refined_context = "\n\n".join(kept_strips).strip()
    return {
        "strips": strips,
        "kept_strips": kept_strips,
        "refined_context": refined_context
    }
    
 