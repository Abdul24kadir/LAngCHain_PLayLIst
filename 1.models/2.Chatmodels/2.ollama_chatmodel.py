from langchain_ollama import ChatOllama

model = ChatOllama(model="gemma3")

result = model.invoke("what is the capital of india")

print(result.content)