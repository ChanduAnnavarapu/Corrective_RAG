from typing import TypedDict,List
from langchain_core.documents import Document
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
    web_docs: List[Document]
    
    refined_context: str
    
    answer: str
    
class DocEvalScore(BaseModel):
    score: float=Field(description="A score between 0 and 1 indicating the relevance of the document to the question.")
    reason: str=Field(description="A short reason explaining the score given to the document.")