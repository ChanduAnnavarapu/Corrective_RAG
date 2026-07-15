from typing import List
import re
from xml.dom.minidom import Document
from state import RAGstate, keepOrDrop, DocEvalScore
from langchain_core.prompts import ChatPromptTemplate
from llm import get_llm
from dotenv import load_dotenv
import os
load_dotenv()

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def refine_context(state: RAGstate)->RAGstate:
    docs = state["Docs"]
    context = "\n\n".join(docs).strip()
    
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
    
 