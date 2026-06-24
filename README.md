# 📄 Finance RAG Copilot: Local GPU Document Analysis

An end-to-end, privacy-first Retrieval-Augmented Generation (RAG) system designed to securely index and query financial documents. 

By leveraging **quantized local Large Language Models (LLMs)** running entirely on a local GPU, this project ensures that sensitive document contents are analyzed without ever sending user prompts to third-party APIs like OpenAI.

## 🚀 Features
- **Data Privacy First:** Uses local LLMs (`microsoft/Phi-3-mini-4k-instruct` or `Gemma`) running on consumer GPU hardware via 4-bit quantization (`bitsandbytes`).
- **Cloud Vector Storage:** Integrates with **Pinecone** for fast, scalable semantic search.
- **Robust RAG Pipeline:** Built with **LlamaIndex**, featuring optimal chunking, overlap, and Hugging Face sentence transformers for embeddings.
- **Interactive UI:** Features a sleek, responsive frontend built with **Streamlit**.
- **Hallucination Prevention:** Implements dynamic stop-tokens and strict prompt templates to prevent the LLM from generating fake conversational loops.

## 🛠️ Architecture & Tech Stack
- **Framework:** LlamaIndex
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** Pinecone (Serverless)
- **Local LLM Inference:** HuggingFace Transformers, PyTorch, Accelerate
- **Frontend UI:** Streamlit

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```
**2. Create a virtual environment and install dependencies**
```bash
conda create -n finance-copilot python=3.10
conda activate finance-copilot
pip install -r requirements.txt
```
**3. Configure Environment Variables**
Create a .env file in the root directory and add your Pinecone API key. (See .env.example for reference).
PINECONE_API_KEY=your_actual_key_here

**4. Add Documents**
Create a data/ folder in the root directory and place your PDF documents inside it.

## 🏃‍♂️ Usage
Step 1: Build the Vector Index
Parse, chunk, embed, and upload your documents to your Pinecone vector database.
python index_builder.py
Step 2: Launch the Application
Start the Streamlit web interface to interact with your documents. (Note: The first launch will take a few moments to download the local LLM weights to your machine).

streamlit run streamlit_app.py

Alternatively, you can run the terminal-based query engine for a pure CLI interaction:
python query_engine.py

## 🧠 Why Local LLMs?
In the financial sector, data privacy is paramount. Standard RAG architectures rely on sending chunks of private documents to commercial APIs (like GPT-4). This project demonstrates how to build a highly capable alternative that keeps all proprietary text within your local memory and network constraints, offloading only the non-sensitive vector embeddings to Pinecone.
