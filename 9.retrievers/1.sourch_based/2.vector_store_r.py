from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langchain_core.documents import Document

documents = [
    Document(page_content="LangChain helps in building LLM applications by connecting language models with external data sources and tools."),

    Document(page_content="LangChain provides components like prompt templates, chains, agents, and memory to simplify AI application development."),

    Document(page_content="Retrieval Augmented Generation (RAG) combines vector databases and LLMs to answer questions using external documents."),

    Document(page_content="Vector stores like Chroma, FAISS, and Pinecone are used to store embeddings and perform similarity search."),

    Document(page_content="Embeddings convert text into numerical vectors so that machines can understand semantic meaning."),

    Document(page_content="Text splitters in LangChain break large documents into smaller chunks for better retrieval and embedding.")
]
embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="generative-ai"
)

retriever = vector_store.as_retriever(search_kwargs={
    'k':2
})

query = 'What is text splitting?'

results = retriever.invoke(query)
print(results)