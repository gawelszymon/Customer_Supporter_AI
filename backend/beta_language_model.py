import os

import pandas as pd
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

base_folder = "./base1"

file_path = "./base/cennik.csv"

reader = SimpleDirectoryReader(base_folder, recursive=True, exclude_hidden=True)

reader.input_files

docs = reader.load_data()

for doc in docs:
    print(doc.get_metadata_str())
    
# docs[0].__dict__      #by that command you can get access to interior directory of object docs[0]

node_parser = SentenceSplitter(chunk_size= 200, chunk_overlap=0) # it parts the data for the chunks that are used to the nodes

nodes = node_parser.get_nodes_from_documents(docs, show_progress=True) # that is the assignment of the document to the nodes

print(len(nodes))