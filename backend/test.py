import os
from llama_index.core import SimpleDirectoryReader

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


base = './exemplary_base'

def get_meta(file_path):
    return {"file_path": os.path.basename(file_path)}

loader = SimpleDirectoryReader(base, required_exts=['.csv', '.txt'], file_metadata=get_meta, recursive=True)
documents = loader.load_data()

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
Settings.text_splitter = text_splitter

model_name_for_embeddings = "BAAI/bge-small-en-v1.5"
embed_model = HuggingFaceEmbedding(model_name=model_name_for_embeddings)

index = VectorStoreIndex.from_documents(
    documents, embed_model=embed_model, transformations=[text_splitter], show_progress=True
) # It simply stores all of the Documents and returns all of them to your query engine.

index.storage_context.persist('./index_storage')

#the alternative way to store index is by using the chromadb

#new document to the index can be added by using insert method

# how to loade the persisted index to avoide reindexing the data:
# # rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
# # load index
# index = load_index_from_storage(storage_context)

retriever = VectorIndexRetriever(index=index, simlarity_top_k=2)

response_synthesizer = get_response_synthesizer()

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

response = query_engine.query("Jaka jest Cena Płyta OSB 12mm z kategorii Materiały Budowlane")
print('='*80)
print(response)