from llama_index.llms.groq import Groq
from getpass import getpass
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
from os import sep
from llama_index.readers.file import PandasCSVReader
from llama_index.core import Document
import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from getpass import getpass
import gradio as gr

GROQ_API_KEY = getpass()

llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

data = pd.read_csv("./cennik.csv", delimiter=";", on_bad_lines="warn")

documents = []
nodes = []
for _, row in data.iterrows():
    metadata = {
        "NAZWA": row["NAZWA"],
        "KATEGORIE": row["KATEGORIE"],
        "PRODUCENT": row["PRODUCENT"],
        "CENA_KATALOGOWA": row["CENA_KATALOGOWA"],
        "CENA_KLIENT": row["CENA_KLIENT"],
        "WALUTA": row["WALUTA"],
        "VAT": row["VAT"],
        "RABAT_KLIENTA": row["RABAT_KLIENTA"],
        "WAGA": row["WAGA"],
        "OPAKOWANIE": row["OPAKOWANIE"],
        "JEDNOSTKA_MIARY": row["JEDNOSTKA_MIARY"],
        "TYP_KURIERA": row["TYP_KURIERA"],
        "KURIERZY_NAZWA": row["KURIERZY_NAZWA"],
        "ACTIVE": row["ACTIVE"],
    }

    # Treść węzła: nazwa i opis produktu
    content = f"{row['NAZWA']}, czyli {row['OPIS']}"
    nodes.append(TextNode(text=content, metadata=metadata))
    
embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-small-en-v1.5")
index=VectorStoreIndex(nodes, embed_model=embed_model)
Settings.embed_model = embed_model

# query_engine=index.as_query_engine(llm=llm)

# response = query_engine.query("Jaki jest najtańszy produkt? Jaką ma nazwę i ile kosztuje?")
# print(response)

def chatbot(input_text):
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(input_text)

    return response

iface = gr.Interface(fn=chatbot, 
                    inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                    outputs=gr.components.Textbox(label="Response"),
                    title="Customer Support AI",
                    allow_flagging="never")

iface.launch(share=True, debug=True)