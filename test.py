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

# print("\nFirst document:")
# print(data["documents"][0])

# print("\nFirst metadata:")
# print(data["metadatas"][0])
print("*"*70)
for i in range(len(data['metadatas'])):
    print(data['documents'][i])
    print(data['metadatas'][i]['source'])
print("*"*70)


data = vector_store.get()

unique_pdfs = set()

for meta in data["metadatas"]:
    if meta and "source" in meta:
        pdf_name = meta["source"]
        unique_pdfs.add(pdf_name)

print("PDFs Found:", unique_pdfs)