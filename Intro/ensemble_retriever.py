# https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4
# https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval/
# https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval/
import os 
os.environ["OPENAI_API_KEY"] = os.environ.get('APIKEY')
print("Initial Done ----------------")


from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Document
reader = SimpleDirectoryReader(input_files=["supervised.pdf"])
docs0 = reader.load_data()
doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]
print('docs',docs)


from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex

llm = OpenAI(model="gpt-3.5-turbo")
chunk_sizes = [1256, 512, 1024]
nodes_list = []
vector_indices = []
for chunk_size in chunk_sizes:
    print(f"Chunk Size: {chunk_size}")
    splitter = SentenceSplitter(chunk_size=chunk_size)
    nodes = splitter.get_nodes_from_documents(docs)

    # add chunk size to nodes to track later
    for node in nodes:
        node.metadata["chunk_size"] = chunk_size
        node.excluded_embed_metadata_keys = ["chunk_size"]
        node.excluded_llm_metadata_keys = ["chunk_size"]

    nodes_list.append(nodes)

    # build vector index
    vector_index = VectorStoreIndex(nodes)
    vector_indices.append(vector_index)

from llama_index.core.tools import RetrieverTool
from llama_index.core.schema import IndexNode

# retriever_tools = []
retriever_dict = {}
retriever_nodes = []
for chunk_size, vector_index in zip(chunk_sizes, vector_indices):
    node_id = f"chunk_{chunk_size}"
    node = IndexNode(
        text=(
            "Retrieves relevant context from supervised.pdf (chunk size"
            f" {chunk_size})"
        ),
        index_id=node_id,
    )
    retriever_nodes.append(node)
    retriever_dict[node_id] = vector_index.as_retriever()

from llama_index.core.selectors import PydanticMultiSelector

from llama_index.core.retrievers import RouterRetriever
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core import SummaryIndex

# the derived retriever will just retrieve all nodes
summary_index = SummaryIndex(retriever_nodes)

retriever = RecursiveRetriever(
    root_id="root",
    retriever_dict={"root": summary_index.as_retriever(), **retriever_dict},
)

nodes = await retriever.aretrieve(
    "Tell me about the main aspects of safety fine-tuning"
)


print(f"Number of nodes: {len(nodes)}")
for node in nodes:
    print(node.node.metadata["chunk_size"])
    print(node.node.get_text())

