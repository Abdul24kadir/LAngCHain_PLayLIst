from langchain_ollama import OllamaEmbeddings


embedding = OllamaEmbeddings(model="all-minilm")

query="india is a asian country"

result = embedding.embed_query(query)
print(result)