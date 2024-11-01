# from sentence_transformers import SentenceTransformer

# sentences = ["this is an this sentence", "Each this is converted"]

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# embeddings = model.encode(sentences)
# print(embeddings)


try:
    from llama_index import Document
    print("Moduł jest dostępny.")
except ModuleNotFoundError:
    print("Moduł nie jest zainstalowany.")
