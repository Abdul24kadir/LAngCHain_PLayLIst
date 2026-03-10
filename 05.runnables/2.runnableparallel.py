from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
from dotenv import load_dotenv
import os
load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_token:
    raise ValueError("HUGGINGFACE API KEY NOT FOUND")

prompt1 = PromptTemplate(
    template="generate a tweet on  {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="generate a linked post on {topic}",
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

parallel_chain = RunnableParallel(
    {
        "tweet":RunnableSequence(prompt1,model,parser),
        "linkedin":RunnableSequence(prompt2,model,parser)
    }
)

result = parallel_chain.invoke({'topic':'ai'})

print("tweet:",result['tweet'])


print("linkedin:",result['linkedin'])