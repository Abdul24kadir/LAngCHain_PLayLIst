from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
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

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )


style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 


length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('C:/1SKILL COMBACK/Langchain/2.prompts/prompt_template.json')

if st.button("summarize"):
    chain = template | model
    result =  chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    st.write(result.content)