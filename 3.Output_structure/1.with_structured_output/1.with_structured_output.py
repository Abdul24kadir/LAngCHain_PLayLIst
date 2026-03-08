# 3 ways :
# 1.TypedDict
# 2.Pydantic 
# 3.Json_schema

# this code will show how to use typedDict

from typing import TypedDict,Annotated,Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token :
    raise ValueError("Api token not found in the .env file")


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    max_new_tokens=1000,
    temperature=0,
    huggingfacehub_api_token=api_token
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    phone : str
    launch_date:str
    processor:str
    battery:str
    pros:Annotated[Optional[list[str]],"this is the battery capacity"]
    cons:str


structured_model = model.with_structured_output(Review)

prompt ="""
Extract phone, launch_date, processor, battery, pros and cons from the text below.
Return only JSON.
Text:
The iQOO 15 is a premium flagship smartphone launched in late 2025, built for users who want top-tier performance, gaming power, and long-term software support. It features Qualcomm’s latest Snapdragon 8 Elite Gen 5 processor, a large and vibrant 6.85-inch LTPO AMOLED display with a smooth 144 Hz refresh rate, and a strong triple-50 MP rear camera system for versatile photography. The phone also comes with a large 7000 mAh battery and fast charging, along with an advanced cooling system to keep performance stable during heavy use. With a promised 5 years of OS updates and 7 years of security patches, the iQOO 15 aims to offer long-lasting value while delivering a flagship-level experience across display quality, speed, camera performance, and battery life.
"""
result = structured_model.invoke(prompt)

print(result)