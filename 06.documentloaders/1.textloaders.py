from langchain_community.document_loaders import TextLoader

loader = TextLoader(r"C:\1SKILL COMBACK\Langchain\6.documentloaders\judgement_24.txt",encoding='utf-8')

docs = loader.load()

print(docs[0].page_content)