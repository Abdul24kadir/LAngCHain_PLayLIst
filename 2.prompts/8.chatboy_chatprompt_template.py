from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import AIMessage,HumanMessage
import streamlit as st
import os
from dotenv import load_dotenv

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_token:
    raise ValueError("api_token not found in the .env file")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens=500,
    task="text-generation",
    huggingfacehub_api_token=api_token,
    temperature=1
)

model = ChatHuggingFace(llm=llm)


template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chathistory=[]
if 'chathistory.txt':
    with open('chathistory.txt') as f:
        chathistory.extend(f.readlines())
else:
    chathistory.save('chathistory.txt')


while True:
    user_input = input('User:')
    prompt = template.invoke({'chat_history':chathistory,'query':user_input})
    if user_input=="exit":
        with open('chathistory.txt','w') as f:
            f.write("\n".join(chathistory))
    chathistory.append(HumanMessage(user_input))
    result = model.invoke(prompt)
    chathistory.append(AIMessage(content=result.content))
    print('AI:',result.content)



print(chathistory)