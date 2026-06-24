import os
import torch
import time  # Import the time module for performance measurement
from dotenv import load_dotenv
from pinecone import Pinecone

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

def main():
    """
    A robust query engine that supports multiple models, provides high-quality
    responses by using model-specific configurations, and measures performance.
    """
    # --- 1. Load Configuration and Select Model ---
    print("Loading configuration...")
    load_dotenv()
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set.")
    
    index_name = "finance-index"
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- CHOOSE YOUR MODEL ---
    # Uncomment the model you want to use.
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    # model_name = "google/gemma-2b-it"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    print(f"Selected model: {model_name}")

    # --- 2. Conditional Model Loading ---
    print("Loading model... (This may take a while depending on the model size)")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)

    # Use an if/else block to handle the Llama 3 offloading requirement
    if "meta" in model_name:
        # For Llama 3, apply the specific memory configuration for CPU offloading
        print("Applying Llama 3 specific memory configuration for CPU offloading.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            max_memory={0: "3.7GiB", "cpu": "7GiB"} # Your working settings
        )
    else:
        # For Phi-3, Gemma, or other models that fit entirely on the GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- 3. DYNAMIC STOP TOKEN CONFIGURATION ---
    # Create a list of stopping tokens that is specific to the selected model
    stop_tokens = ["\n"] # A newline is a good general-purpose stop token
    if "Llama-3" in model_name:
        stop_tokens.append("<|eot_id|>")
    elif "Phi-3" in model_name:
        stop_tokens.append("<|end|>")
    elif "gemma" in model_name:
        stop_tokens.append("<eos>")

    # Get the actual token IDs from the tokenizer
    stopping_ids = [
        tokenizer.convert_tokens_to_ids(token) for token in stop_tokens if token in tokenizer.get_vocab()
    ]
    
    # Also add the generic end-of-sequence token if it's not already in our list
    if tokenizer.eos_token_id not in stopping_ids:
        stopping_ids.append(tokenizer.eos_token_id)
    
    print(f"Using stopping tokens: {stop_tokens}")

    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=4096,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        stopping_ids=stopping_ids,
    )
    Settings.llm = llm

    # --- 4. Define a Strong Prompt Template ---
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

    # --- 5. Connect to Vector DB and Build Query Engine ---
    print("Connecting to vector DB and building query engine...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine(text_qa_template=qa_template)

    print(f"\n✅ GPU Query engine with {model_name} is ready.")
    print("Type 'exit' or 'quit' to stop.")

    # --- 6. Interactive Query Loop with Performance Timing ---
    while True:
        try:
            query = input("\nAsk a question or give a command: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            print(f"Thinking... (Using {model_name})")
            
            # --- TIMER START ---
            start_time = time.time()
            
            response = query_engine.query(query)
            
            # --- TIMER END ---
            end_time = time.time()
            duration = end_time - start_time
            
            print("\n--- Response ---")
            print(str(response).strip())
            print("\n--- Sources ---")
            for node in response.source_nodes:
                print(f"- Source: {node.metadata.get('file_name', 'N/A')}, Score: {node.score:.4f}")
            
            # --- PRINT THE TIME TAKEN ---
            print(f"\n--- Time Taken: {duration:.2f} seconds ---")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    main()