import os
from llama_index.core import SimpleDirectoryReader

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings


base = './exemplary_base'
base_embedding_model = 'thenlper/gte-base'

def get_meta(file_path):
    return {"file_path": os.path.basename(file_path)}

loader = SimpleDirectoryReader(base, required_exts=['.csv', '.txt'], file_metadata=get_meta, recursive=True)
documents = loader.load_data()

print(f'type:\t {type(documents)}')
print(f'len:\t  {len(documents)}')
print(f'doc[0]:\t {type(documents[0])}')


model_name_for_embeddings = "BAAI/bge-small-en-v1.5"
embed_model = HuggingFaceEmbedding(model_name=model_name_for_embeddings)
embedding = embed_model.get_query_embedding("any text")
print(f'embedding (vector): {embedding}')
print(f'len(vector): {len(embedding)}')
Settings.chunk_size = 512
Settings.chunk_overlap = 25
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=True)

retriever = VectorIndexRetriever(index=index, simlarity_top_k=2)

prompt_text = "Płyta OSB 12mm?"
fragments = retriever.retrieve(prompt_text)

print(f'number of text fragments: {len(fragments)}')
print('='*80)

for fragment in fragments:
    print(f'metadata: {fragment.metadata}')
    print(f'score: {fragment.get_score()}')
    print(f'text ({len(fragment.text)}): {fragment.text}')
    print('='*80)
    
model_name_for_generation = 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf'

# model for generation
# other LLMs: https://docs.llamaindex.ai/en/stable/examples/llm/openllm/
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

response_iter = llm.stream_complete("Jaka jest Cena Płyta OSB 12mm z kategorii Materiały Budowlane")

for response in response_iter:
    print(response.delta, end="", flush=True)
    
query_engine = index.as_query_engine(similarity_top_k=2, llm=llm)

prompts = query_engine.get_prompts()
print(prompts['response_synthesizer:text_qa_template'].get_template())

response = query_engine.query("Jaka jest Cena Płyta OSB 12mm z kategorii Materiały Budowlane")
print('='*80)
print(response)