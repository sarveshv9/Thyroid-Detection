import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)

class ThyroidBot:
    def __init__(self, knowledge_base_dir='knowledge_base', persist_directory='chroma_db'):
        load_dotenv()
        
        # Check API key
        if "GOOGLE_API_KEY" not in os.environ:
            logger.warning("GOOGLE_API_KEY not found in environment variables. Chatbot may not work.")
            
        self.knowledge_base_dir = knowledge_base_dir
        self.persist_directory = persist_directory
        self.vector_store = None
        self.qa_chain = None
        
        # Initialize Embeddings (Local HuggingFace model)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize LLM (Gemini)
        # Using gemini-flash-latest as a default capable and fast model
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
        
        # System Prompt
        system_prompt = (
            "You are a helpful, professional AI assistant for a Thyroid Disease Detection application. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "If you don't know the answer, just say that you don't know. "
            "Do not provide medical diagnoses, always advise consulting a healthcare professional for actual medical advice.\n\n"
            "Context: {context}"
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
    def initialize(self):
        """Loads documents, creates vector store, and initializes the chain."""
        try:
            logger.info("Initializing RAG pipeline...")
            
            # 1. Load text documents
            txt_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.txt", loader_cls=TextLoader)
            txt_documents = txt_loader.load()
            
            # Load PDF documents
            pdf_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            pdf_documents = pdf_loader.load()
            
            documents = txt_documents + pdf_documents
            
            if not documents:
                logger.warning(f"No documents found in {self.knowledge_base_dir}")
                return False
                
            # 2. Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # 3. Create or load vector store
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # 4. Create chains
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            logger.info("RAG pipeline initialized successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            return False
            
    def ask(self, question: str) -> str:
        """Query the RAG pipeline."""
        if not self.qa_chain:
            return "The chatbot is not fully initialized or missing an API key. Please check the server logs."
            
        try:
            response = self.qa_chain.invoke({"input": question})
            return response.get("answer", "I'm sorry, I couldn't generate a response.")
        except Exception as e:
            logger.error(f"Error during chatbot query: {e}")
            return "An error occurred while processing your request. Please try again later."
