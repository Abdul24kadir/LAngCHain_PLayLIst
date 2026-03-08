from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os 


load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("hugging face api token is not found in the .env file")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=api_token,
    max_new_tokens=512,
    temperature=0.2,
)

model = ChatHuggingFace(llm = llm)


st.header("Base Paper Summarize")

user_input = st.text_input("Enter Your Prompt")
if st.button("summarize"):
    result = model.invoke(user_input)
    st.write(result.content)

#basic representation of summarize model using user prompts .
#what are flaws in this :
#1.cannot use code 
#2.cannot assign role 
#3.cannot store chat history if a chatbot is made 
#4.the prompt is static 
