from xml.dom.minidom import Document
from state import RAGstate, DocEvalScore
from langchain_core.prompts import ChatPromptTemplate
from llm import get_llm
from typing import List
import os
from dotenv import load_dotenv
import re
load_dotenv()

   
def eval_doc_node(state: RAGstate)->RAGstate:
    docs = state["Docs"]
    question = state["Question"]
    
    doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
    )

    doc_eval_chain = doc_eval_prompt | get_llm().with_structured_output(DocEvalScore)
    good_docs = []
    scores: List[float] = []
    reasons: List[str] = []
    good: List[Document] = []
    UPPER_TH = float(os.getenv("UPPER_TH"))
    LOWER_TH = float(os.getenv("LOWER_TH"))
    
    if len(docs) > 0:
        for doc in docs:
            out=doc_eval_chain.invoke({"question": question, "chunk": doc})
            scores.append(out.score)
            reasons.append(out.reason)
            
            if out.score > LOWER_TH:
                good.append(doc)
    
    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    # 3) INCORRECT if all docs < LOWER_TH
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        why = "No chunk was sufficient."
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}. {why}",
        }

    # 4) Anything in between => AMBIGUOUS
    why = "Mixed relevance signals."
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. {why}",
    }
    