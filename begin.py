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
    documents, transformation=[SentenceSplitter(chunk_size=1028)]
)



# 4. Create an index (this will use your local embedding model)
#index = VectorStoreIndex.from_documents(documents)

# 5. Create a query engine (this will use your local Phi-3 model)
query_engine = index.as_query_engine()

# 6. Ask a question!
response = query_engine.query(" IIT Computer Science major preparing for an off-campu what is forst step ? and what chucks were retreived")

# 7. Print the response
print(response)

'''
Reads all files from the "data" folder (could be .txt, .pdf, etc.).

Splits them into Document objects (internally into Nodes, with metadata).

You should learn: How chunking happens, what metadata is preserved, and why chunk size matters.
'''


