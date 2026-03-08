from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_token:
    raise ValueError("HUGGINGFACE API KEY NOT FOUND")

prompt1 = PromptTemplate(
    template="Tell me a joke on {topic}",
    input_variables=['topic']
)

end_point= HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=api_token,
    task="text-generation",
    max_new_tokens=700
)

model = ChatHuggingFace(llm = end_point)

parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser)

result = chain.invoke({"artificial intelligence"})

print(result)