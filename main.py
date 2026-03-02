# main code for rag

# step-1 imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# step-2 load website


# step-3 text splitting


# step-4 create and store embeddings


# step-5 create retriever


# step-6 llm (local)


# step-7 prompt


# step-8 format retrieved docs


# step-9 rag pipeline


# step-10 ask question


# end of code