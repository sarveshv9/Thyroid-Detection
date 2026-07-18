import os
import shutil
import time
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def init_db():
    print("Removing old chroma_db...")
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        
    print("Loading documents...")
    knowledge_base_dir = "knowledge_base"
    txt_loader = DirectoryLoader(knowledge_base_dir, glob="**/*.txt", loader_cls=TextLoader)
    txt_documents = txt_loader.load()
    
    pdf_loader = DirectoryLoader(knowledge_base_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    
    documents = txt_documents + pdf_documents
    print(f"Loaded {len(documents)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Total chunks: {len(splits)}")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    batch_size = 80 # below the 100 limit
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        print(f"Adding batch {i//batch_size + 1}/{(len(splits)-1)//batch_size + 1}...")
        vector_store.add_documents(batch)
        if i + batch_size < len(splits):
            print("Sleeping for 60 seconds to avoid rate limits...")
            time.sleep(60)
            
    print("Done building new chroma_db!")

if __name__ == "__main__":
    init_db()
