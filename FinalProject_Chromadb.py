from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import pandas as pd
import os 

from langchain_ollama import OllamaEmbeddings

# Project documents path

folders = [
    r'C:\Users\Angelin\OneDrive - The University of Colorado Denver\Angelin Tisha Classes\Computing for BA\Final Project\Knowledge\4-Year_Plans_2024-25',
    r'C:\Users\Angelin\OneDrive - The University of Colorado Denver\Angelin Tisha Classes\Computing for BA\Final Project\Knowledge\Business_Course_Catalog_Descriptions', 
    r'C:\Users\Angelin\OneDrive - The University of Colorado Denver\Angelin Tisha Classes\Computing for BA\Final Project\Knowledge\Degree Plan Templates',
    r'C:\Users\Angelin\OneDrive - The University of Colorado Denver\Angelin Tisha Classes\Computing for BA\Final Project\Knowledge\Fall_2005_Class_List',
    r'C:\Users\Angelin\OneDrive - The University of Colorado Denver\Angelin Tisha Classes\Computing for BA\Final Project\Knowledge\Minors'
      ]
# Function to load excel using panda and convert to text and then to document
def loadExcel(filePath):

    docs = []

    xls = pd.ExcelFile(filePath)

    for sheet in xls.sheet_names:
        df = pd.read_excel(filePath, sheet_name=sheet)
        df = df.dropna(how="all")   # remove empty rows
        text = df.to_csv(index=False)

        doc = Document(
            page_content=text,
            metadata={
                "source": filePath,
                "sheet": sheet,
                "filetype": "excel"
            }
        )
        docs.append(doc)

    return docs

# Load all documents
allDocs = [] 

for folder in folders:

    files = os.listdir(folder)
    
    for file in files:
            # Skip Office temporary files like ~$file.ext
            if file.startswith("~$"):
                print("Skipping temp file:", file)
                continue

            filePath = os.path.join(folder, file)

            if file[-5:] == ".docx":
                loader = Docx2txtLoader(filePath)
                docs = loader.load()
                for d in docs:
                    d.metadata["filetype"] = "docx"
                    d.metadata["source"] = filePath
                    d.metadata["section"] = os.path.basename(folder)
                allDocs.extend(docs)

            elif file[-4:] == ".pdf":
                loader = PyPDFLoader(filePath)
                docs = loader.load()
                for d in docs:
                    d.metadata["filetype"] = "pdf"
                    d.metadata["source"] = filePath
                    d.metadata["section"] = os.path.basename(folder)
                allDocs.extend(docs)
            
            elif file[-5:] == ".xlsx":
                docs = loadExcel(filePath)
                allDocs.extend(docs)
            
            else:
                print("Skipped file:", file)

print("Documents loaded:", len(allDocs))

# Text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)

chunks = splitter.split_documents(allDocs)

print("Chunks created:", len(chunks))

# Text Embeddings
embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

# Creates a sqlite database called vectorstore.db
vectorstore = Chroma.from_documents(
    chunks, 
    embedding=embedding_function, 
    persist_directory=r'C:\Users\Angelin\OneDrive - The University of Colorado Denver\Angelin Tisha Classes\Computing for BA\Final Project\data\chroma_advising_db'
)

vectorstore

# Similarity Search
result = vectorstore.similarity_search("What is BANA 6620?", k = 3)
len(result)
print(result[0].page_content)
