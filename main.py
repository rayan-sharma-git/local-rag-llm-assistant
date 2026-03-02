# main code for rag

# step-1 imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# step-2 load website
loader = WebBaseLoader(
    'https://python.langchain.com/docs/introduction/'
)

documents = loader.load()
print('Documents loaded: ',len(documents))

# step-3 text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = text_splitter.split_documents(documents)

# step-4 create and store embeddings
embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

index_path = 'faiss_index'

if os.path.exists(index_path):
    print('Loading existing FAISS index...')
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print('Creating new FAISS index...')
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

# step-5 create retriever
retriever = vectorstore.as_retriever(
    search_kwargs = {'k':3}
)

# step-6 llm (local)
llm = OllamaLLM(model = 'tinyllama')

# step-7 prompt
template = '''
You are a helpful assistant.

Answer ONLY using the context below.
If answer not in context, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}
'''

prompt = ChatPromptTemplate.from_template(template)

# step-8 format retrieved docs
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# step-9 rag pipeline
rag_chain = (
    {
        'context': retriever | format_docs,
        'question': lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)

# step-10 ask question
while True:
    query = input('\nAsk a question (or type "exit"): ')

    if query.lower() == 'exit':
        break

    response = rag_chain.invoke(query)
    print('\nAnswer:\n')
    print(response)

# end of code