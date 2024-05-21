import os 
os.environ["OPENAI_API_KEY"] = os.environ.get('APIKEY')
print("Initial Done ----------------")

### Creating Llamaindex Documents ###
## word does not have pages info
from llama_index.core import SimpleDirectoryReader
reader = SimpleDirectoryReader(input_files=["lorem.pdf"])
documents = reader.load_data()
print(documents)