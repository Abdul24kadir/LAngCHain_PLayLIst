from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
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
    chathistory.append(user_input)
    if user_input == 'exit':
        break
    result = model.invoke(chathistory)
    chathistory.append(result.content)
    print("Ai:",result.content)
# the model failed because it doesnot had the chat history
# user:which is greater 3 or 2?
# Ai:  Three is greater than two. In mathematical terms, the relationship between the two numbers is expressed as 3 > 2. This means that the value of the number 3 is greater than the value of the number 2.
# user:multiply the greater number with 10.
# Ai:  To multiply the greater number by 10, you first need to identify which number is the greater one. Let's call it "number A". Here's how to perform the multiplication:

# Number A * 10

# For example, if Number A is 5, then:

# 5 * 10 = 50

# Or if Number A is 12, then:

# 12 * 10 = 120
