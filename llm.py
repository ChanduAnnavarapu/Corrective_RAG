from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    llm = ChatGroq(model="llama-3.3-70b-versatile", 
                    api_key=os.getenv("API_KEY"),
                    temperature=0.1)
    return llm


def llm_invoke(prompt):
    llm = get_llm()
    response = llm.invoke(prompt)
    return response


def get_embeddings():
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )
    return embeddings