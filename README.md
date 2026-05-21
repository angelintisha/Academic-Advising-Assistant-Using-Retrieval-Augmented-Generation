# Academic Advising Assistant (RAG-Based)
> A conversational AI assistant that answers student academic advising questions 
> using Retrieval-Augmented Generation (RAG), LangChain, and a locally hosted LLM.

## Overview
An AI-powered advising chatbot built for a university business school that retrieves 
accurate answers from institutional documents — degree plans, course catalogs, and 
class schedules — without hallucinating information outside the source documents.

## Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-LLM-gray)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-blue)

## How It Works
1. **Document Ingestion** — Loads institutional PDF, Word, and Excel files using LangChain loaders
2. **Chunking** — Splits documents using Recursive Character Text Splitter (chunk size: 900, overlap: 150)
3. **Embeddings** — Converts chunks to vector embeddings using `mxbai-embed-large` via Ollama
4. **Vector Store** — Stores and retrieves embeddings using ChromaDB for semantic search
5. **RAG Pipeline** — Retrieves top-k relevant chunks and passes them as context to the LLM
6. **LLM Response** — `llama3.2` (locally hosted via Ollama) generates grounded answers
7. **Chat Interface** — Deployed as an interactive chat app using Streamlit

## Example Questions
| Question | Source Document |
|----------|----------------|
| What are the core courses in Marketing? | marketing_4yp_2024-25.docx |
| What are the graduation requirements for Accounting? | accounting_4yp_2024-25.docx |
| What is the course information for BANA 6620? | graduate_business_course_descriptions.docx |
| What are the time and location details for BANA 6620 in Fall 2025? | Fall 2025 Class List_ZW.xlsx |
| What courses does Ziyi Wang teach in Fall 2025? | Fall 2025 Class List_ZW.xlsx |
| What is the prerequisite for BANA 6620? | graduate_business_course_descriptions.docx |
| What courses should a freshman in Marketing take in Semester 1? | marketing_4yp_2024-25.docx |

## Key Design Decisions
- **Grounded responses only** — LLM is strictly prompted to answer from retrieved context only, preventing hallucination
- **Query enhancement** — User questions are automatically expanded with keywords before retrieval to improve chunk matching
- **Source attribution** — Every response includes the source document it was retrieved from
- **Local LLM** — Runs entirely on-device via Ollama; no external API calls or data sharing

## Limitations & Future Work
- Structured Excel tables lose formatting when converted to text, which can affect accuracy
- Future improvements: database storage for scheduling data, refined chunking for tabular content

## Files
| File | Description |
|------|-------------|
| `FinalProject_Chromadb.py` | Document ingestion, chunking, embedding, and vector store creation |
| `FinalProject_Retriever.py` | RAG chain setup and sample query testing |
| `FinalProject_StreamlitApp.py` | Full Streamlit chat application |
| `Report.pdf` | Full project report |
