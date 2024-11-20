import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')

def get_meta(file_path):
    return{"file_path": os.base}

base_folder = "./exemplary_base"
reader = SimpleDirectoryReader(base_folder, file_metadata=get_meta ,recursive=True, exclude_hidden=True)
reader.input_files
docs = reader.load_data()
nodes_parser = SentenceSplitter(chunk_size=200, chunk_overlap=0)
nodes = nodes_parser.get_nodes_from_documents(docs, show_progress=True)
index = VectorStoreIndex(nodes, embed_model=embed_model)
index_dir = "./index_storage"
index.storage_context.persist(index_dir)