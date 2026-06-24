import os
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def main():
    """
    Builds or updates a Pinecone index.
    - Creates the index if it doesn't exist.
    - Adds new documents to the index if it already exists.
    """
    # --- 1. Load Configuration ---
    print("Loading configuration...")
    load_dotenv()
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set.")
    
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = None
    index_name = "finance-index"

    # --- 2. Initialize Pinecone and Vector Store ---
    print("Initializing Pinecone connection...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Index '{index_name}' already exists.")

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # --- 3. Load New Documents ---
    print("Loading new documents from './data'...")
    documents = SimpleDirectoryReader("./data").load_data()
    if not documents:
        print("No new documents to add. Exiting.")
        return
    print(f"✅ Loaded {len(documents)} new document sections.")

    # --- 4. The CORRECT Way to Update an Existing Index ---
    # Load the index object from the existing vector store
    print("Loading existing index from vector store...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Create a node parser with our desired chunk settings
    node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)

    # Use the .insert_documents() method to add the new data
    print("Inserting new documents into the index...")
    nodes = node_parser.get_nodes_from_documents(documents)
    index.insert_nodes(nodes, show_progress=True)
    
    print("\n✅ Index update complete.")

if __name__ == "__main__":
    main()