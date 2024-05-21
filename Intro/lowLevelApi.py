#https://docs.llamaindex.ai/en/stable/examples/callbacks/TokenCountingHandler/#token-counting-handler
### Low-Level Query API ###

### Initial ###
import os 
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, TokenCountingHandler
import tiktoken
os.environ["OPENAI_API_KEY"] = os.environ.get('APIKEY')
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = CallbackManager([llama_debug,token_counter])
print("Initial Done ----------------")

### Creating Llamaindex Documents ###
from llama_index.core import download_loader
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
wikipedia_documents = loader.load_data(pages=['Iceland Country','Kenya Country','Cambodia Country'])
# print(wikipedia_documents)
print("Creating Llamaindex Documents Done----------------")


### Creating LlamaIndex Nodes ###
# from llama_index.core.node_parser import SimpleNodeParser
# parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
# wikipedia_nodes = parser.get_nodes_from_documents(wikipedia_documents)
# print("len(wikipedia_nodes)",len(wikipedia_nodes))#49
# print("Creating LlamaIndex Nodes Done----------------")

### Using Index to Query Data ####
import nest_asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
# from llama_index.core import ServiceContext
from llama_index.core.service_context import ServiceContext

nest_asyncio.apply()

# # Log the API usage




# We are using the LlamaDebugHandler to print the trace of the sub questions captured by the SUB_QUESTION callback event type

# DeprecationWarning

# service_context = ServiceContext()
vector_query_engine = VectorStoreIndex.from_documents(
    wikipedia_documents, use_async=True
).as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="countries",
            description="Wikipedia pages about the countries - Iceland, Kenya, Cambodia.",
        ),
    ),
]
print('total',token_counter.total_embedding_token_count)
token_counter.reset_counts()
print('###')


query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)


print('###')
response_english = query_engine.query(
    "Give me all similaries between Iceland, Kenya and Cambodia"
)
# response_english All three countries, Iceland, Kenya, and Cambodia, have unique geographical features that contribute to their diverse landscapes. Additionally, each country has a rich cultural heritage that includes traditional art forms and historical backgrounds marked by significant events and influences.
print("response_english",response_english)
print('embedding count',token_counter.total_embedding_token_count)
print('prompt count',token_counter.prompt_llm_token_count)
print('completion count',token_counter.completion_llm_token_count)
print('total count',token_counter.total_llm_token_count)
token_counter.reset_counts()

print('###')
response_chinese = query_engine.query(
    "請告訴我冰島、肯亞和柬埔寨的人口數"
)
# response_chinese 冰島的人口數約為32萬人，肯亞的人口數超過4760萬人，柬埔寨的人口數約為1672萬人。
print("response_chinese",response_chinese)
print('embedding count',token_counter.total_embedding_token_count)
print('prompt count',token_counter.prompt_llm_token_count)
print('completion count',token_counter.completion_llm_token_count)
print('total count',token_counter.total_llm_token_count)
print('###')