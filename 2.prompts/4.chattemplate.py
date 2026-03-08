from langchain_core.prompts import ChatPromptTemplate

chat_template =ChatPromptTemplate([
    ('system','You are helpful {domain} assistant'),
    ('human','tell me about {query}')
])
prompt = chat_template.invoke({'domain':'cricket','query':'Leg Before Wicket'})

print(prompt)