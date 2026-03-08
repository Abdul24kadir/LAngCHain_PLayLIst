from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_community.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import JsonOutputParser 
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

schema = [
    ResponseSchema(name="most_runs", description="integer value of most runs"),
    ResponseSchema(name="most_centuries", description="integer value of most centuries"),
    ResponseSchema(name="odi_centuries", description="integer value of odi centuries"),
]

parser = StructuredOutputParser.from_response_schema(schema)

template1 = PromptTemplate(
    template="""
    give the achivements of {name} as per 2025 
    this achievement should have 
    1.Most runs 
    2.Most centuries
    3.Odi centuries
    \n {format_instructions}
    """, 
    input_variables=["name"],
    partial_variables={"format_instructions":parser.get_format_instructions()},
)

chain = template1 | model | parser

result = chain.invoke({'name':'Virat KOhli'})

print(result)
