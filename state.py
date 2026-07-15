from typing import TypedDict,List
from pydantic import BaseModel,Field

from chromadb import Documents

class RAGstate(TypedDict):
    Question: str
    Docs:List[str]
    
    good_docs: List[str]
    verdict: str
    reason: str
    
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    
    answer: str
    
class keepOrDrop(BaseModel):
    keep: bool=Field(description="True if the sentence is relevant to the question, False otherwise.")
    
class DocEvalScore(BaseModel):
    score: float=Field(description="A score between 0 and 1 indicating the relevance of the document to the question.")
    reason: str=Field(description="A short reason explaining the score given to the document.")