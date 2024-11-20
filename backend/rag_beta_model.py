import os
from llama_index.core import SimpleDirectoryReader


base = './exemplary_base'
base_embedding_model = 'thenlper/gte-base'

def get_meta(file_path):
    return {"file_path": os.path.basename(file_path)}

loader = SimpleDirectoryReader(base, required_exts=['.csv', '.txt'], file_metadata=get_meta, recursive=True)
documents = loader.load_data()

print(f'type:\t {type(documents)}')
print(f'len:\t  {len(documents)}')
print(f'doc[0]:\t {type(documents[0])}')

print(documents[1])