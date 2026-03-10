from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')

query = "the geopolitical history of india and pakistan from the perspective of china"

docs = retriever.invoke(query)

print("Number of docs:", len(docs))

for i, doc in enumerate(docs):
    print(f"\n--- result {i+1} ---")
    print(doc.page_content[:500])