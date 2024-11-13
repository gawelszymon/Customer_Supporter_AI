from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core.llms import LLM, CompletionResponse, ChatResponse
import torch
from typing import List, Dict, Any
from pydantic import Field, PrivateAttr

class CustomHuggingFaceLLM(LLM):
    model_name: str = Field(description="neme of hugging face model")
    tokenizer_name: str = Field(description="name of the hugging face tokenizer")
    device_map: str = Field(default="auto", description="Device map for model loading")
    cache_dir: str = Field(default="./model_cache", description="Directory to store model weights")
    offload_dir: str = Field(default="./model_offload", description="Directory for weight offloading")
    context_window: int = Field(default=2048, description="maximum context window size (max quantity of input tokens)")
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

