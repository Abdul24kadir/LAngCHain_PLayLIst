from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path=r"C:\1SKILL COMBACK\Langchain\6.documentloaders\sample.csv")
docs = loader.load()
print(docs[0].page_content)