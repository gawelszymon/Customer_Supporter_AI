from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import pandas as pd

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

tariff_file = 'cennik.csv'
df = pd.read_csv(tariff_file, delimiter=';')

passages = df['NA_PROMOCJI'].tolist()
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", passages=passages)

print(df.head(10))
