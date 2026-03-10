# note : hugging face models may or may not support pydantic parser even your code structure is correct
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os       
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")   
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")    

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=api_token,
    max_new_tokens=450,
    temperature=0.3,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)    

class Review(BaseModel):
    mostruns : int = Field(description="runs scored")
    mostcenturies : int = Field(description="total centuries")
    odicenturies : int = Field(description="total odi centuries")

parser = PydanticOutputParser(pydantic_object=Review)

template1 = PromptTemplate(
    template="""
    give the achivements of {name} as per year 2025 
    \n {format_instructions}
    """, 
    input_variables=["name"],
    partial_variables={"format_instructions":parser.get_format_instructions()},
)

chain = template1 | model | parser

result = chain.invoke({'name':'Virat KOhli'})

print(result)
