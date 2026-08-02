from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma,FAISS
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

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


def get_all_pdfs():
    try:
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=get_embeddings()
        )

        data = vector_store.get()

        unique_pdfs = set()

        for meta in data["metadatas"]:
            if meta and "source" in meta:
                pdf_name = os.path.basename(meta["source"])
                unique_pdfs.add(pdf_name)

        #print("PDFs Found:", unique_pdfs)

        return sorted(unique_pdfs)

    except Exception as e:
        print(e)
        return []
    
    
def delete_embeddings_of_book_from_vectorstore(book_name):
    vector_store = Chroma(
    embedding_function=get_embeddings(),
    persist_directory="./chroma_db"
    )

    data = vector_store.get()
    found = False

    for metadata in data["metadatas"]:
        if metadata["source"] == book_name:
            found = True
            break

    if found:
        vector_store.delete(where={"source": book_name})
        print(f"Deleted embeddings for {book_name}")
    else:
        print(f"No embeddings found for {book_name}")
        
        
if __name__ == "__main__":
    #create_vector_store(filepath="uploaded_docs/Leave-Policy.pdf")
    #get_all_pdfs()
    delete_embeddings_of_book_from_vectorstore("uploaded_docs\\Mahi_book.pdf")