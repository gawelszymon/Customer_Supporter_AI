from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from llama_index.core.schema import TextNode


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


nodes = [
    TextNode(
        text="Kosiarka spalinowa",
        metadata={
            "marka": "Stiga",
            "cena": 1200,
            "stan magazynowy": 15,
            "kategoria": "Narzędzia ogrodnicze"
        }
    ),
    TextNode(
        text="Podkaszarka elektryczna",
        metadata={
            "marka": "Bosch",
            "cena": 300,
            "stan magazynowy": 25,
            "kategoria": "Narzędzia ogrodnicze"
        }
    ),
    TextNode(
        text="Sekator do gałęzi",
        metadata={
            "marka": "Fiskars",
            "cena": 70,
            "stan magazynowy": 40,
            "kategoria": "Narzędzia ogrodnicze"
        }
    ),
    TextNode(
        text="Ręczny opryskiwacz",
        metadata={
            "marka": "Gardena",
            "cena": 45,
            "stan magazynowy": 50,
            "kategoria": "Narzędzia ogrodnicze"
        }
    ),
    TextNode(
        text="Grabie metalowe",
        metadata={
            "marka": "Wolf-Garten",
            "cena": 30,
            "stan magazynowy": 35,
            "kategoria": "Narzędzia ogrodnicze"
        }
    ),
    TextNode(
        text="Pistolet do kleju",
        metadata={
            "marka": "Yato",
            "cena": 70,
            "stan magazynowy": 22,
            "kategoria": "Narzędzia wykończeniowe"
        }
    ),
    TextNode(
        text="Taśma malarska",
        metadata={
            "marka": "Tesa",
            "cena": 10,
            "stan magazynowy": 100,
            "kategoria": "Narzędzia wykończeniowe"
        }
    ),
    TextNode(
        text="Szpachla stalowa",
        metadata={
            "marka": "Condor",
            "cena": 20,
            "stan magazynowy": 55,
            "kategoria": "Narzędzia wykończeniowe"
        }
    ),
    TextNode(
        text="Wałek malarski",
        metadata={
            "marka": "Hamilton",
            "cena": 35,
            "stan magazynowy": 30,
            "kategoria": "Narzędzia wykończeniowe"
        }
    ),
    TextNode(
        text="Paca z zębami",
        metadata={
            "marka": "Hardex",
            "cena": 15,
            "stan magazynowy": 40,
            "kategoria": "Narzędzia wykończeniowe"
        }
    ),
    TextNode(
        text="Wkrętarka akumulatorowa",
        metadata={
            "marka": "Makita",
            "cena": 500,
            "stan magazynowy": 20,
            "kategoria": "Narzędzia warsztatowe"
        }
    ),
    TextNode(
        text="Szlifierka kątowa",
        metadata={
            "marka": "Bosch",
            "cena": 250,
            "stan magazynowy": 18,
            "kategoria": "Narzędzia warsztatowe"
        }
    ),
    TextNode(
        text="Młotek ślusarski",
        metadata={
            "marka": "Stanley",
            "cena": 40,
            "stan magazynowy": 45,
            "kategoria": "Narzędzia warsztatowe"
        }
    ),
    TextNode(
        text="Klucz nastawny",
        metadata={
            "marka": "Neo Tools",
            "cena": 30,
            "stan magazynowy": 60,
            "kategoria": "Narzędzia warsztatowe"
        }
    ),
    TextNode(
        text="Imadło warsztatowe",
        metadata={
            "marka": "Topex",
            "cena": 150,
            "stan magazynowy": 10,
            "kategoria": "Narzędzia warsztatowe"
        }
    )
]


model_name_for_embeddings = "BAAI/bge-small-en-v1.5"
embed_model = HuggingFaceEmbedding(model_name=model_name_for_embeddings)
index = VectorStoreIndex(nodes, embed_model=embed_model)
index_dir = "./index_storage"
index.storage_context.persist(index_dir)

retriever = VectorIndexRetriever(index=index, simlarity_top_k=2)

model_name_for_generation = 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf'
llm = LlamaCPP(model_url=model_name_for_generation,
            temperature=0.7,
            max_new_tokens=256,
            context_window=4096, # 4096 is max for Llama2
            generate_kwargs = {"stop": ["<s>", "[INST]", "[/INST]"]},# kwargs for Llama: stop==A list of strings to stop generation when encountered.
            # other kwargs for generation: top_k, top_p, min_p, frequentcy_penalty, repeat_penalty,
            model_kwargs={"n_gpu_layers": 43},  # for GPU acceleration (nuber of layers for GPU offloading)
            # other kwargs for model: lora_path, lora_base
            verbose=True)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

query_engine = index.as_query_engine(similarity_top_k=2, llm=llm)
response = query_engine.query("Jaka jest Cena Imadła Warsztatowego marki Topex?")
print('='*80)
print(response)