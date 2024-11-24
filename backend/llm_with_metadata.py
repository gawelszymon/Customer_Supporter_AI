from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# index_storage = "index_storage"


storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage"),
    vector_store=SimpleVectorStore.from_persist_dir(
        persist_dir="./storage"
    ),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage"),
)

index = load_index_from_storage(storage_context)

# retriever = VectorIndexRetriever(index=index, simlarity_top_k=2)

# model_name_for_generation = 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf'
# llm = LlamaCPP(model_url=model_name_for_generation,
#             temperature=0.7,
#             max_new_tokens=256,
#             context_window=4096, # 4096 is max for Llama2
#             generate_kwargs = {"stop": ["<s>", "[INST]", "[/INST]"]},# kwargs for Llama: stop==A list of strings to stop generation when encountered.
#             # other kwargs for generation: top_k, top_p, min_p, frequentcy_penalty, repeat_penalty,
#             model_kwargs={"n_gpu_layers": 43},  # for GPU acceleration (nuber of layers for GPU offloading)
#             # other kwargs for model: lora_path, lora_base
#             verbose=True)
# Settings.llm = llm
# Settings.embed_model = embed_model
# Settings.chunk_size = 512

# query_engine = index.as_query_engine(similarity_top_k=2, llm=llm)
# response = query_engine.query("Jaka jest Cena Imadła Warsztatowego marki Topex?")
# print('='*80)
# print(response)