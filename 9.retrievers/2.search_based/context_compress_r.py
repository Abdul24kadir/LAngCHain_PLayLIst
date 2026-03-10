from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from dotenv import load_dotenv
import os


# Load API token
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
# Create documents
docs = [
    Document(page_content="Photosynthesis is the process by which green plants use sunlight to synthesize food."),
    Document(page_content="Plants convert carbon dioxide and water into glucose and oxygen."),
    Document(page_content="Machine learning is a field of artificial intelligence that uses statistical techniques."),
]

# Create FAISS vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embedding_model)

# Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Setup LLM
llm_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=api_token,
    max_new_tokens=200,
    temperature=0.3,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm_endpoint)

# Create compressor
compressor = LLMChainExtractor.from_llm(model)

# Create contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Query
query = "What is photosynthesis?"

compressed_results = compression_retriever.invoke(query)

# Print results
for doc in compressed_results:
    print(doc.page_content)