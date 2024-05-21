# https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4
# https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval/
# https://docs.llamaindex.ai/en/stable/module_guides/indexing/index_guide/
# https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/
import os 
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.schema import IndexNode
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core import SummaryIndex
from llama_index.core import Settings
os.environ["OPENAI_API_KEY"] = os.environ.get('APIKEY')
print("Initial Done ----------------")



reader = SimpleDirectoryReader(input_files=["supervised.pdf"])
docs0 = reader.load_data()
doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]
# print('docs',docs)




llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = CallbackManager([llama_debug])

llm = OpenAI(model="gpt-3.5-turbo")
# chunk_sizes = [1256, 512, 1024]
chunk_sizes = [256,512]
# nodes_list = []
# # vector_indices = []
# for chunk_size in chunk_sizes:
#     print(f"Chunk Size: {chunk_size}")
#     splitter = SentenceSplitter(chunk_size=chunk_size)
#     nodes = splitter.get_nodes_from_documents(docs)

#     # add chunk size to nodes to track later
#     for node in nodes:
#         node.metadata["chunk_size"] = chunk_size
#         node.excluded_embed_metadata_keys = ["chunk_size"]
#         node.excluded_llm_metadata_keys = ["chunk_size"]

#     nodes_list.append(nodes)

#     # build vector index
#     vector_index = VectorStoreIndex(nodes)
#     folder_name = "SUPERVISED{}".format(chunk_size)
#     vector_index.storage_context.persist(persist_dir=folder_name)
#     # vector_indices.append(vector_index)




storage_context256 = StorageContext.from_defaults(persist_dir="SUPERVISED256")
storage_context512 = StorageContext.from_defaults(persist_dir="SUPERVISED512")
vector_indices = [load_index_from_storage(storage_context256),load_index_from_storage(storage_context512)]

####################################################### insert
# # retriever_tools = []
# retriever_dict = {}
# retriever_nodes = []
# for chunk_size, vector_index in zip(chunk_sizes, vector_indices):
#     node_id = f"chunk_{chunk_size}"
#     node = IndexNode(
#         text=(
#             "Retrieves relevant context from supervised.pdf (chunk size"
#             f" {chunk_size})"
#         ),
#         index_id=node_id,
#     )
#     retriever_nodes.append(node)
#     retriever_dict[node_id] = vector_index.as_retriever()

# summary_index = SummaryIndex(retriever_nodes)
####################################################### insert
####################################################### load
# the derived retriever will just retrieve all nodes
# summary_index = SummaryIndex(retriever_nodes)
# summary_index.storage_context.persist("SUPERVISEDSUMMARY")


retriever_dict = {}
for chunk_size, vector_index in zip(chunk_sizes, vector_indices):
    node_id = f"chunk_{chunk_size}"
    node = IndexNode(
        text=(
            "Retrieves relevant context from supervised.pdf (chunk size"
            f" {chunk_size})"
        ),
        index_id=node_id,
    )
    retriever_dict[node_id] = vector_index.as_retriever()

storage_context = StorageContext.from_defaults(persist_dir="SUPERVISEDSUMMARY")
summary_index = load_index_from_storage(storage_context)
print("summary_index",summary_index)
print("summary_index.as_retriever()",summary_index.as_retriever())
####################################################### load
retriever = RecursiveRetriever(
    root_id="root",
    retriever_dict={"root": summary_index.as_retriever(), **retriever_dict},
)


query_engine = RetrieverQueryEngine(retriever)

response = query_engine.query(
    "i want to know Taiwan population"
)
print('response',response)

response = query_engine.query(
    "where is taiwan"
)
print('response',response)

response = query_engine.query(
    " List all the Supervised Learning in the documents"
)
print('response',response)