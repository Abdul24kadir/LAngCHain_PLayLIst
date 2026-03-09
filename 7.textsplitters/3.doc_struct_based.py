from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
text = """
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel , Field
from typing import Literal 

from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")        

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=api_token,
    max_new_tokens=450,
    temperature=0.3,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser1 = StrOutputParser()

class sentiment(BaseModel):
    sentiment:Literal['Positive','Negative'] = Field (description="sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=sentiment)

Sentimentprompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into Positive or Negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Write an appropriate response to this Positive feedback \n {feedback}'",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this Negative feedback \n {feedback}'",
    input_variables=['feedback']
)

conditional_chain = RunnableBranch(
    (lambda x:x.sentiment =='Positive',prompt2|model|parser1),
    (lambda x:x.sentiment =='Negative',prompt3|model|parser1),
    RunnableLambda(lambda x:"could not find sentiment")
)

sentiment_chain = Sentimentprompt1 | model | parser2

chain = sentiment_chain |conditional_chain

feedback = "the phone is wonderful"

result = chain.invoke({'feedback':feedback})

print(result)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size = 100,
    language = Language.PYTHON,
    chunk_overlap = 0
)
result = splitter.split_text(text)
print(result)