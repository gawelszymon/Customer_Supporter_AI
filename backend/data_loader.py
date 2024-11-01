import pandas as pd
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

tariff_file = 'cennik.csv'
df = pd.read_csv(tariff_file, delimiter=';')

lakierobejce = df[df['NAZWA'].str.contains("lakierobejca", case=False)]

lakierobejce['CENA_KLIENT'] = pd.to_numeric(lakierobejce['CENA_KLIENT'], errors='coerce')

passages = (lakierobejce['NAZWA'] + " kosztuje " + lakierobejce['CENA_KLIENT'].astype(str) + " PLN").tolist()

retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", passages=passages)

query = "Jaka jest cena LAKIEROBEJCA ZEWNĘTRZNA 2W1 ORZECH CIEMNY 0.2L"
input_ids = tokenizer(query, return_tensors="pt").input_ids

generated = model.generate(input_ids, num_beams=2, min_length=10, max_length=50)
output = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(output[0])