from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="feature-extraction"
)

documents = [
    "virat kohli is the king of cricket",
    "sachin tendulkar is one the gratest batsman in cricket history",
    "rohit sharma has all trophies except world cup",
    "jasprit bumrah is a rising indian pacer",
    "dhoni was captian of indian odi team in 2011"
]

query = 'tell me about virat kohli?'
doc_embedding = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embedding)[0]
index , score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(documents[index])
print("similarity scores:",score)