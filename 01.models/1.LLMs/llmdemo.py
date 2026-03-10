from langchain_ollama import OllamaLLM
from langchain_openai import OpenAI
from dotenv import load_dotenv
# load_dotenv()
# llm = OpenAI(model = "gpt-3.5-turbo-instance")
llm = OllamaLLM(model="llama3.2")
result = llm.invoke('what is the capital of india')
print(result)