# https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4
# https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval/

# import os 
# os.environ["OPENAI_API_KEY"] = os.environ.get('APIKEY')
# print("Initial Done ----------------")


# from llama_index.core import SimpleDirectoryReader
# from llama_index.core.node_parser import SimpleNodeParser
# reader = SimpleDirectoryReader(input_files=["supervised.pdf"])
# docs = reader.load_data()
# node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
# base_nodes = node_parser.get_nodes_from_documents(docs)


# sub_chunk_sizes = [128, 256, 512]
# sub_node_parsers = [
#     SimpleNodeParser.from_defaults(chunk_size=c) for c in sub_chunk_sizes
# ]

# all_nodes = []
# for base_node in base_nodes:
#     for n in sub_node_parsers:
#         sub_nodes = n.get_nodes_from_documents([base_node])
#         sub_inodes = [
#             IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
#         ]
#         all_nodes.extend(sub_inodes)

#     # also add original node to node
#     original_node = IndexNode.from_text_node(base_node, base_node.node_id)
#     all_nodes.append(original_node)
# all_nodes_dict = {n.node_id: n for n in all_nodes}