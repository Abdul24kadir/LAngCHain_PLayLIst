from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("there is no api key in .env file")


chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=api_token,
    max_new_tokens=512,
    temperature=0.2,
)
model = ChatHuggingFace(llm = llm)

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)