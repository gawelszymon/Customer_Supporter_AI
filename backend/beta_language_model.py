import os

import pandas as pd
from llama_index.core import SimpleDirectoryReader

base_folder = "./base"

file_path = "./base/cennik.csv"

reader = SimpleDirectoryReader(base_folder, recursive=True, exclude_hidden=True)

reader.input_files

docs = reader.load_data()

for doc in docs:
    print(doc.get_metadata_str())