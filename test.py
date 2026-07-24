from langchain_community.vectorstores import Chroma
from llm import get_embeddings
import os

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=get_embeddings()
)

data = vector_store.get()

print("Keys:", data.keys())
print("Number of documents:", len(data["documents"]))
print("Number of metadata records:", len(data["metadatas"]))

print("\nFirst document:")
print(data["documents"][0])

print("\nFirst metadata:")
print(data["metadatas"][0])


data = vector_store.get()

unique_pdfs = set()

for meta in data["metadatas"]:
    if meta and "source" in meta:
        pdf_name = os.path.basename(meta["source"])
        unique_pdfs.add(pdf_name)

print("PDFs Found:", unique_pdfs)