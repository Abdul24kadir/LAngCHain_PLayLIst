from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
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

prompt2 = PromptTemplate(
    template="explain this {text}",
    input_variables=['text']
)


end_point= HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=api_token,
    task="text-generation",
    max_new_tokens=700
)

model = ChatHuggingFace(llm = end_point)

parser = StrOutputParser()
joke = RunnableSequence(prompt1,model,parser)
chain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "explain":RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke,chain)
result = final_chain.invoke({'topic':'ai'})

print(result)