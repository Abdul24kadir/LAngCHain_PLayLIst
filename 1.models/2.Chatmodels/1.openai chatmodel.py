from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="")#use chatopenai model
result = model.invoke("what is capital of India?")
print(result.content)
