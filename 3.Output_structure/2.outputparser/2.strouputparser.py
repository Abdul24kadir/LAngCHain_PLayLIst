from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")        

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    huggingfacehub_api_token=api_token,
    max_new_tokens=450,
    temperature=0.3,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1=PromptTemplate(
    template="write a detailed biography of {name}",
    input_variables=["name"]
)

template2 = PromptTemplate(
    template="give the 5 achivements: {text}",
    input_variables=["text"]
)

chain = template1 | model | parser | template2| model | parser 

result = chain.invoke({'name':'Virat Kohli'})

print(result)