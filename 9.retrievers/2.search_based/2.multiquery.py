# MultiQuery Retriever Example

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os


# Load API token
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")


documents = [

    Document(
        page_content="LangChain is a framework used to build applications powered by large language models. It helps developers connect LLMs with tools, APIs, and external data."
    ),

    Document(
        page_content="Large Language Models (LLMs) like GPT, Llama, and Claude are capable of generating human-like text and answering questions."
    ),

    Document(
        page_content="Retrieval Augmented Generation (RAG) improves LLM responses by retrieving relevant documents from a vector database before generating an answer."
    ),

    Document(
        page_content="Vector databases such as FAISS, Chroma, Pinecone, and Weaviate store embeddings and allow similarity search over large collections of text."
    ),

    Document(
        page_content="Embeddings transform text into numerical vectors so machines can understand semantic relationships between sentences."
    ),

    Document(
        page_content="Prompt engineering is the process of designing prompts to guide the behavior of large language models and improve their outputs."
    )

]


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)


llm_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=api_token,
    max_new_tokens=200,
    temperature=0.3,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm_endpoint)


similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    llm=model
)



query = "What are embeddings in AI?"



sim_results = similarity_retriever.invoke(query)
mul_results = multi_query_retriever.invoke(query)



print("\n========== SIMILARITY RETRIEVER ==========")

for i, doc in enumerate(sim_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


print("\n========== MULTI QUERY RETRIEVER ==========")

for i, doc in enumerate(mul_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)