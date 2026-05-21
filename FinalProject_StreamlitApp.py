# conda activate bana
# cd "C:\Users\Angelin\OneDrive - The University of Colorado Denver\Angelin Tisha Classes\Computing for BA\Final Project"
# streamlit run FinalProject_StreamlitApp.py

# Libraries

from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import os

# Initialize the Streamlit app
st.set_page_config(page_title="Academic Advising Assistant", layout="wide")
st.title("Academic Advising Assistant")

with st.expander("Example Questions"):
    st.markdown(
        """
        1. What are the core courses in Marketing?  
        2. What are the graduation requirements for Accounting?  
        3. What is the course information for BANA 6620?  
        4. What are the time and location details for BANA 6620 in Fall 2025?  
        5. What courses does Ziyi Wang teach in Fall 2025, and what is his email address?  
        6. What is the prerequisite for BANA 6620?  
        7. What courses should a freshman in Marketing take in Semester 1?
        """
    )

# Improve query
def improve_query(question):
    q = question.lower().strip()

    if "core courses" in q:
        return question + " exact 4-year plan Core Classes Business Core Marketing Required Classes course list only"

    elif "graduation requirements" in q:
        return question + " exact GENERAL GRADUATION REQUIREMENTS & POLICIES bullet list only"

    elif "course information" in q:
        return question + " exact course entry full course description restriction grading basis"

    elif "time and location" in q:
        return question + " exact class list row only start date end date start time end time building room"

    elif "teach" in q and "email" in q:
        return question + " exact instructor row only course title email no raw row"

    elif "prerequisite" in q:
        return question + " exact course prereq only"

    elif "freshman" in q or "semester 1" in q:
        return question + " exact SAMPLE ACADEMIC PLAN OF STUDY Year One Semester 1 course list only"

    else:
        return question
        
# Format retrieved docs before sending to model
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Keep only unique file names, max 1
def unique_source_name(docs, max_source=1):
    seen = set()
    names = []

    for d in docs:
        fileName = os.path.basename(d.metadata.get("source", ""))

        if fileName and fileName not in seen:
            seen.add(fileName)
            names.append(fileName)

        if len(names) >= max_source:
            break

    return names

# Function to create the processing chain
def create_chain():
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma(
        persist_directory=r'C:\Users\Angelin\OneDrive - The University of Colorado Denver\Angelin Tisha Classes\Computing for BA\Final Project\data\chroma_advising_db', embedding_function=embedding_function
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    template = """You are an academic advising assistant.

Answer the question using ONLY the context below.

Rules:
- Use only information directly stated in the context.
- Do not combine information from different courses, rows, or documents.
- Do not add commentary, explanation, notes, or assumptions.
- Do not repeat raw row text, CSV-style text, or unrelated lines.

- If the question asks for core courses in a degree program:   return only the course IDs and course titles from the 4-year plan and do not include credits, descriptions, prerequisites, restrictions, or electives.

- If the question asks for graduation requirements: answer only from the GENERAL GRADUATION REQUIREMENTS & POLICIES section and do not add any note before or after the bullets.

- If the question asks for course information, use only the exact course entry and include:
  - course title
  - credits
  - full course description
  - prerequisite if present
  - restriction if present
  - grading basis if present

- If the question asks specifically for prerequisites, extract only the prerequisite line for that exact course, if no prerequisite is explicitly listed for that course, say "No prerequisite present for this course'.

- If the question asks about time and location for a course, use only the exact matching class-list row and return exactly these items, each on its own line:
  Start Date:
  End Date:
  Start Time:
  End Time:
  Location:
  Building:
  Room:
  Do not include instructor, email, or any other fields.
  
- Format dates as DD-Mon-YYYY.
- Format times as HH:MM, with no seconds.

- If the question asks what courses an instructor teaches:
  use only rows containing that exact instructor name.
  Return only:
  Course titles:
  - one course title per line
  Instructor email:
  - email on its own line
  Do not include raw row text, dates, times, or other fields.

- If the question asks what a freshman should take in Semester 1:
  answer only from the Sample Academic Plan of Study.
  Use only Year One, Semester 1.
  Return only the course IDs and course titles, one per line.
  Do not include courses from other semesters.
  Do not include descriptions, prerequisites, credits, or electives.

- If the answer truly is not in the context, say exactly:
  "I don't have that information in the provided documents."

Context:
{context}

Question: {question}

Answer:
"""

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="llama3.2")
    chain = (
        {"context": retriever | format_docs, 
         "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain, retriever

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! Ask me an academic advising question."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Enter your question here:"):
    
        # Add user message to chat history
        with st.chat_message("user"):
            st.markdown(question)

        # Save user message
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):

            # Get chain and retriever
            chain, retriever = create_chain()

            # Retrieve source documents
            better_question = improve_query(question)
            docs = retriever.invoke(better_question)
            
            # Get the response from the AI model
            response = chain.invoke(better_question)

            # Keep only 1 unique source file names
            source_name = unique_source_name(docs, max_source=1)

            # Add source directly below answer
            if source_name:
                response += f"\n\n**Source:**\n- {source_name[0]}"

         # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})