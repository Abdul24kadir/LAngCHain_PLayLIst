from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("there is no api key in .env file")

chathistory = []

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=api_token,
    max_new_tokens=512,
    temperature=0.2,
)

model = ChatHuggingFace(llm = llm)

while True:
    user_input = input("user:")
    chathistory.append(HumanMessage(user_input))
    if user_input == 'exit':
        print(chathistory)
        break
    result = model.invoke(chathistory)
    chathistory.append(AIMessage(result.content))
    print("Ai:",result.content)

