from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


embeddings = OllamaEmbeddings(model="nomic-embed-text")

db_location = "/home/jithu/Downloads/LocalAIAgentWithRAG-main/chrome_langchain_db2"
add_documents = not os.path.exists(db_location)

reader = PdfReader("pytorch.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200
)
docs = splitter.create_documents([text])
print(f"Created {len(docs)} chunks")
#print(docs)
        
vector_store = Chroma(
    collection_name="pdf_docs1",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=docs)
    #vector_store.persist()
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
query = "What were the sales figures for Q1?"
results = Chroma.similarity_search(query, k=5)
for r in results:
    print(f"[{r.metadata['type']}] {r.page_content[:200]}…\n")