from pathlib import Path
import gradio as gr
import sys
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.llms import LLM, CompletionResponse, ChatResponse
from llama_index.core.prompts.prompts import SimpleInputPrompt
import torch
from typing import Optional, List, Dict, Any
from pydantic import Field, PrivateAttr
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP



class CustomHuggingFaceLLM(LLM):
    model_name: str = Field(description="The name of the HuggingFace model to use")
    tokenizer_name: str = Field(description="The name of the HuggingFace tokenizer to use")
    device_map: str = Field(default="auto", description="Device map for model loading")
    cache_dir: str = Field(default="./model_cache", description="Directory to store model weights")
    offload_dir: str = Field(default="./model_offload", description="Directory for weight offloading")
    context_window: int = Field(default=2048, description="Maximum context window size")
    max_new_tokens: int = Field(default=256, description="Maximum number of new tokens to generate")
    
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _device: Any = PrivateAttr()
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, cache_dir=self.cache_dir)
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            cache_dir=self.cache_dir,
            offload_folder=self.offload_dir,
            torch_dtype=torch.float16
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        outputs = self._model.generate(
            inputs["input_ids"],
            max_new_tokens=256,
            temperature=0.7,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        response_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response_text[len(prompt):].strip()
        
        return CompletionResponse(text=response_text)
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> ChatResponse:
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        completion_response = self.complete(prompt, **kwargs)
        return ChatResponse(message=completion_response.text)
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        return self.complete(prompt, **kwargs)
    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponse:
        return self.complete(prompt, **kwargs)
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)
    async def astream_complete(self, prompt: str, **kwargs) -> CompletionResponse:
        return self.complete(prompt, **kwargs)
    async def astream_chat(self, messages: List[Dict[str, str]], **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)
    @property
    def metadata(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "device": self._device.type,
            "context_window": self.context_window,
            "max_new_tokens": self.max_new_tokens
        }
# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
docs_path = "./exemplary_base"
embed_model = 'thenlper/gte-base'

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""
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

def initialize_index():
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print(f"Created directory: {docs_path}")
        return None
    
    documents = SimpleDirectoryReader(docs_path).load_data()
    if not documents:
        print("No documents found in the specified directory.")
        return None
        
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# Initialize the index
index = initialize_index()
def chatbot(input_text):
    if index is None:
        return "Error: No documents found in the specified directory.", []
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    
    relevant_files = []
    for node_with_score in response.source_nodes:
        file = node_with_score.node.metadata['file_name']
        full_file_path = Path(docs_path, file).resolve()
        
        if full_file_path not in relevant_files:
            relevant_files.append(full_file_path)
    
    return response.response, relevant_files
# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.components.Textbox(lines=7, label="Enter your text"),
    outputs=[
        gr.components.Textbox(label="Response"),
        gr.components.File(label="Relevant Files")
    ],
    title="Custom-trained AI Chatbot",
    allow_flagging="never"
)
if __name__ == "__main__":
    iface.launch(share=False)