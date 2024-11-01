from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceLLM
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
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
            
        return documents

    def _create_product_description(self, row):
        description_parts = [
            f"Nazwa produktu: {row['NAZWA']}",
            f"Indeks: {row['INDEKS']}",
            f"Producent: {row['PRODUCENT']}",
            f"Kategorie: {row['KATEGORIE']}",
            f"EAN: {row['EAN']}" if pd.notna(row['EAN']) else "",
            f"Kod producenta: {row['KOD_PRODUCENTA']}" if pd.notna(row['KOD_PRODUCENTA']) else "",
            f"Opis: {row['OPIS']}" if pd.notna(row['OPIS']) else "",
            f"Krótki opis: {row['KROTKI_OPIS']}" if pd.notna(row['KROTKI_OPIS']) else "",
            f"Extra opis: {row['EXTRA_OPIS']}" if pd.notna(row['EXTRA_OPIS']) else "",
            f"Cena katalogowa: {row['CENA_KATALOGOWA']} {row['WALUTA']}" if pd.notna(row['CENA_KATALOGOWA']) else "",
            f"Jednostka miary: {row['JEDNOSTKA_MIARY']}" if pd.notna(row['JEDNOSTKA_MIARY']) else "",
            f"Waga: {row['WAGA']}" if pd.notna(row['WAGA']) else "",
            f"Opakowanie: {row['OPAKOWANIE']}" if pd.notna(row['OPAKOWANIE']) else ""
        ]
        
        return "\n".join([part for part in description_parts if part])

    def _prepare_metadata(self, row):
        metadata = {}
        for key, value in row.items():
            if pd.notna(value):
                metadata[key] = value
            else:
                metadata[key] = None
                
        return {
            'indeks': metadata.get('INDEKS', None),
            'kategorie': metadata.get('KATEGORIE', None),
            'producent': metadata.get('PRODUCENT', None),
            'ean': metadata.get('EAN', None),
            'na_magazynie': metadata.get('NA_MAGAZYNIE', None),
            'stan_magazynowy': metadata.get('STAN_NA_MAGAZYNIE', None),
            'cena_katalogowa': metadata.get('CENA_KATALOGOWA', None),
            'waluta': metadata.get('WALUTA', None),
            'symbol_kategorii': metadata.get('SYMBOL_KATEGORII', None),
            'aktywny': metadata.get('ACTIVE', None)
        }
        

    def create_index(self):
        documents = self.prepare_documents()
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        
        vector_store = FaissVectorStore(dim=768)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        self.index = VectorStoreIndex(
            nodes,
            service_context=self.service_context,
            storage_context=storage_context
        )
        
        self.index.storage_context.persist(persist_dir=self.persist_dir)

    def load_index():
        pass

    def query():
        pass