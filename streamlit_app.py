import streamlit as st
import os
import torch
from dotenv import load_dotenv

# All the imports from your previous script
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from pinecone import Pinecone

# --- Core App Configuration ---
st.set_page_config(page_title="DocBot", page_icon="🤖", layout="centered")
st.title("📄 DocBot: Your Document Q&A Assistant")
st.write("Ask questions about the documents you have indexed. This bot uses a local LLM on your GPU to answer.")

# --- Model and Index Loading (with Caching) ---
# This is the most important function. The @st.cache_resource decorator
# tells Streamlit to run this function only once, and cache the result.
# This means your slow models are only loaded the first time the app runs.
@st.cache_resource
def load_and_setup():
    """Load all necessary models and set up the connection to Pinecone."""
    load_dotenv()
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        st.error("PINECONE_API_KEY is not set in your environment. Please add it to your .env file.")
        st.stop()
    
    # Configure models
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    # model_name = "google/gemma-2b-it"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    
    with st.spinner("Loading LLM... This may take a moment."):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_config, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    stop_tokens = ["<|end|>", "\n"]
    stopping_ids = [tokenizer.convert_tokens_to_ids(token) for token in stop_tokens if token in tokenizer.get_vocab()]
    stopping_ids.append(tokenizer.eos_token_id)

    llm = HuggingFaceLLM(
        model=model, tokenizer=tokenizer, context_window=4096, max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        stopping_ids=stopping_ids,
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    # Connect to Pinecone and load the index
    index_name = "finance-index"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    return index

# --- Main App Logic ---
try:
    index = load_and_setup()

    # Set up the prompt template
    qa_template_str = (
        "You are an expert analyst. Use the following context to answer the question or fulfill the request.\n"
        "Provide a clear, concise, and direct response based only on the information provided.\n"
        "After providing the answer, you must stop. Do not generate any follow-up questions or answers.\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "Request: {query_str}\n"
        "---------------------\n"
        "Answer: "
    )
    qa_template = PromptTemplate(qa_template_str)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! What can I help you with today?"}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query_engine = index.as_query_engine(text_qa_template=qa_template)
                response = query_engine.query(prompt)
                
                # Format the response with sources
                response_text = str(response).strip()
                sources = [node.metadata.get('file_name', 'N/A') for node in response.source_nodes]
                # Remove duplicate sources
                unique_sources = sorted(list(set(sources)))
                
                full_response = f"{response_text}\n\n*Sources: {', '.join(unique_sources)}*"
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

except Exception as e:
    st.error(f"An error occurred: {e}")