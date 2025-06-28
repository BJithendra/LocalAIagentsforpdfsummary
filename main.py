# main.py
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from csvembedding import retriever

def main():
    # Build or load retriever

    # Initialize your chat LLM
    llm = OllamaLLM(model="qwen3:1.7b", base_url="http://localhost:11434")

    # Prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert. Query the Chroma DB first and give only the answers the user asks only using available information. Search the whole db for asked question before answer"),
        ("human", "{context}"),
        ("human", "{input}")
    ])
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)

    # Retrieval QA chain
    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_chain
    )

    # Interactive loop
    while True:
        query = input("You: ")
        if query.strip().lower() in ("q", "quit", "exit"):
            break
        result = qa_chain.invoke({"input": query})
        print("Bot:", result["answer"])

if __name__ == "__main__":
    main()
