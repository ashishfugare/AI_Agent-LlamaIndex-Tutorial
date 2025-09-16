from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

#default for chucking 
Settings.chunk_size = 512 # new this "I want to parse my documents into smaller chunks

#local Seetiings





# 1. Set up the local models
# Use your downloaded Phi-3 model through Ollama
llm = Ollama(model="gemma:2b", request_timeout=120.0) # Changed model and increased timeout

# Use a local model for creating embeddings
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Configure LlamaIndex to use these local models
Settings.llm = llm
Settings.embed_model = embed_model

# 3. Load your data from the 'data' directory
documents = SimpleDirectoryReader("./data").load_data()


index = VectorStoreIndex.from_documents(
    documents, transformations=[SentenceSplitter(chunk_size=1028)]
)

##"I want to retrieve more context when I query"#



# 4. Create an index (this will use your local embedding model)
#index = VectorStoreIndex.from_documents(documents)

# 5. Create a query engine (this will use your local Phi-3 model)
# The fix is to add streaming=True
# 1. Create a STANDARD query engine for regular Q&A and checking source nodes
query_engine_standard = index.as_query_engine()

# 2. Create a SEPARATE STREAMING engine for real-time responses
query_engine_streaming = index.as_query_engine(streaming=True)

# --- NOW USE THE CORRECT ENGINE FOR EACH TASK ---

# Use the standard engine to get a full response and check the chunks
print("--- Retrieving Chunks ---")
response = query_engine_standard.query("What is the first step for an IIT CS major...")
print(f"LLM Response: {response}\n")
for node in response.source_nodes:
    print(f"Score: {node.score:.4f} | Content: {node.text[:100]}...")

# Use the streaming engine for a real-time response
print("\n--- Streaming Response ---")
streaming_response = query_engine_streaming.query("What is the next step for off-campus prep?")
for token in streaming_response.response_gen:
    print(token, end="", flush=True)
print()

'''
Reads all files from the "data" folder (could be .txt, .pdf, etc.).

Splits them into Document objects (internally into Nodes, with metadata).

You should learn: How chunking happens, what metadata is preserved, and why chunk size matters.
'''


