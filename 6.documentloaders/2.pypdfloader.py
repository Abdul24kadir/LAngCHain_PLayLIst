from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader(r"C:\1SKILL COMBACK\Langchain\6.documentloaders\KADIR._RESUME.pdf")
docs = loader.load()
print(len(docs))
print(docs[0].page_content)