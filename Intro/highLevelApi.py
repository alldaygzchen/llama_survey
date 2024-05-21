### High-Level Query API ###


### Initial ###
import os 
os.environ["OPENAI_API_KEY"] = os.environ.get('APIKEY')
print("Initial Done ----------------")

### Creating Llamaindex Documents ###
from llama_index.core import SimpleDirectoryReader
reader = SimpleDirectoryReader(input_files=["bcg-2022-annual-sustainability-report-apr-2023.pdf"])
pdf_documents = reader.load_data()
# page
# print('len(pdf_documents)',len(pdf_documents))
# print("pdf_documents[0]",pdf_documents[0])
# print("pdf_documents[103]",pdf_documents[103])
print("Creating Llamaindex Documents Done----------------")

### Creating LlamaIndex Nodes ###
from llama_index.core.node_parser import SimpleNodeParser
parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)
pdf_nodes = parser.get_nodes_from_documents(pdf_documents)
# print("len(pdf_nodes)",len(pdf_nodes))
# print("pdf_nodes[0]",pdf_nodes[0])
# print("pdf_nodes[103]",pdf_nodes[103])
print("Creating LlamaIndex Nodes Done----------------")

### Creating LlamaIndex Index ###
# from llama_index.core import VectorStoreIndex
# index = VectorStoreIndex(pdf_nodes)
# # Persisting to disk
# index.storage_context.persist(persist_dir="HIGHLEVELDATA")

from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="HIGHLEVELDATA")
index = load_index_from_storage(storage_context)
print("index",index)
# print("len(index)",len(index))
print("Creating LlamaIndex Index Done----------------")

### Using Index to Query Data ###
query1 = "in what content is Morocco mentioned in the report"
query2= "List measures taken to address diseases occuring in developing industries"
query_engine = index.as_query_engine()
response = query_engine.query(query1)
print("response",response)
response = query_engine.query(query2)
print("response",response)
print("Using Index to Query Data Done----------------")