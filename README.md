# Local RAG LLM Assistant

This is a fully local Retrieval-Augmented Generation (RAG) assistant built using LangChain, Ollama, and FAISS.

The goal of this project is to demonstrate how a complete RAG pipeline works end-to-end using only local resources. The system retrieves relevant document chunks from a FAISS vector store and generates grounded responses using a locally hosted LLM via Ollama.

Everything runs locally after the model is downloaded. No external API keys are required.

---

## What This Project Does

- Loads documents (web or local files)
- Splits text into chunks with overlap
- Generates embeddings
- Stores vectors using FAISS
- Retrieves top-k relevant chunks
- Injects context into the prompt
- Generates grounded answers using a local LLM

---

## Architecture Overview

User Query  
↓  
FAISS Vector Search  
↓  
Top Relevant Chunks  
↓  
Prompt Construction  
↓  
Local LLM (Ollama)  
↓  
Final Answer  

---

## Tech Stack

- Python 3.10.6  
- LangChain  
- Ollama  
- FAISS (CPU)  
- HuggingFace Embeddings  
- BeautifulSoup4  

---

## Clone the Repository

```bash
git clone https://github.com/rayan-sharma-git/local-rag-llm-assistant.git
cd local-rag-llm-assistant
```

---

## Python Version

This project is built using Python 3.10.6.

The version is specified inside the `.python-version` file for consistency.

---

## If You Already Have a Different Python Version Installed

You do NOT need to change your global Python version.

You can install Python 3.10.6 separately and create a virtual environment specifically for this project.

### Step 1: Install Python 3.10.6

Download it from the official Python website and install it normally.

During installation, make sure to check:
"Add Python to PATH"

---

### Step 2: Create a Virtual Environment Using Python 3.10.6

On Windows:

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

On Mac/Linux (if python3.10 is installed):

```bash
python3.10 -m venv venv
source venv/bin/activate
```

This will create an isolated environment using Python 3.10.6 without affecting your system Python.

---

## Install Dependencies

After activating the virtual environment:

```bash
pip install -r requirements.txt
```

---

## Install Ollama and Pull a Model

Install Ollama from its official website, then run:

```bash
ollama pull tinyllama
```

---

## Run the Application

```bash
python main.py
```

Example:

```
Ask a question: What is LangChain?

Answer:
LangChain is a framework for building LLM-powered applications...
```

---

## Project Structure

```
local-rag-llm-assistant/

main.py
requirements.txt
.python-version
vectorstore/   (generated at runtime, non commit)
README.md
.gitignore
```

---

## Notes

- The `vectorstore` folder is created automatically after the first run.
- The project runs entirely locally after model download.