from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma,FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from utils import decompose_to_sentences, refine_context
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()

from state import RAGstate
from llm import get_llm, get_embeddings

def create_vector_store(filepath):
    #Pdf loading
    docs=PyPDFLoader(filepath).load()
    
    # text splitting
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    
    #vector store
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vector_store


def load_vector_store():
    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vector_store

def get_relevant_docs(state:RAGstate):
    query = state["Question"]
    vector_store = load_vector_store()
    relevant_docs = vector_store.similarity_search(query, k=3)
    return { 
            "Docs": [doc.page_content for doc in relevant_docs if len(relevant_docs) > 0]
            }

if __name__ == "__main__":
    create_vector_store(filepath="Leave-Policy.pdf")
