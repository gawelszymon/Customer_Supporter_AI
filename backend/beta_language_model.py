from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceLLM
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from typing import List, Dict
from llama_index import Document


import pandas as pd

class ProductsCatalgoRAG:
    def __init__(self, csv_path: str, persist_dir: str = './storage'):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
    
    def setup_components(self):
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"    #model which convert input sentence into a numbers, model use 758 dimensions to 
        )                                                           #represent each sentence, this kind of mechanism might by use to display 
                                                                    #the similarity between the sentences during searching process
                                                                
        llm = HuggingFaceLLM(
            model_name="google/flan-t5-large",      #generating text model
            tokenizer_name="google/flan-t5-large",      #convert the input text into tokens that are processing by the model
            max_length=512,     #max length of tokens in a generated answer
            temperature=0.1   #control of model's creativity, the value closer to zero the model is more accurate according to data
        )
        
        self.service_context = ServiceContext.from_defaults(  #it's like the engine of our model both for generating the new text by LLM and creating the embed model
            llm=llm,
            embed_model=embed_model,
            chunk_size=256,     #that's snippet of text converted in one fragment by the system
            chunk_overlap=20   #number of the tokens which overlap between the fragments, to provide the coherence and continuity
        )
        

    def prepare_documents(self) -> List[Document]:  #the function returns list of the object with objects of type Document
        df = pd.read_csv(self.csv_path, sep=';', encoding='utf-8')
        documents = []
        
        for idx, row in df.iterrows():
            main_text = self._create_product_description(row)
            metadata = self._prepare_metadata(row)
            
        doc = Document(text=main_text, metadata=metadata)
        documents.append(doc)

    def _create_product_description():
        pass

    def _prepare_metadata():
        pass

    def create_index():
        pass

    def load_index():
        pass

    def query():
        pass