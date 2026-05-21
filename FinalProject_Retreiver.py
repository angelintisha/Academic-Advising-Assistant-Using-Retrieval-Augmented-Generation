# Libraries

from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# 1.0 CREATE A RETRIEVER FROM THE VECTORSTORE

embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

vectorstore = Chroma(
    persist_directory=r'C:\Users\Angelin\OneDrive - The University of Colorado Denver\Angelin Tisha Classes\Computing for BA\Final Project\data\chroma_advising_db',
    embedding_function=embedding_function,
)

retriever = vectorstore.as_retriever()

retriever

# 2.0 USE THE RETRIEVER TO AUGMENT AN LLM

# * Prompt template

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# * LLM Specification
model = Ollama(model="llama3.2")

# * Combine with Lang Chain Expression Language (LCEL)
#   - Context: Give it access to the retriever
#   - Question: Provide the user question as a pass through from the invoke method
#   - Use LCEL to add a prompt template, model spec, and output parsing

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# * Try it out:
result = rag_chain.invoke(
    "What are the core courses in Marketing?"
)

print(result)

result = rag_chain.invoke(
    "What are the graduation requirements for Accounting?"
)

print(result)

result = rag_chain.invoke(
    "What is the course information for BANA 6620?"
)

print(result)

result = rag_chain.invoke(
    "What are the time and location details for BANA 6620 in Fall 2025?"
)

print(result)

result = rag_chain.invoke(
    "What courses does Ziyi Wang teach in Fall 2025, and what is his email address?"
)

print(result)

result = rag_chain.invoke(
    "What is the prerequisite for BANA 6620?"
)

print(result)

result = rag_chain.invoke(
    "What courses should a freshman in Marketing take in Semester 1?"
)

print(result)

