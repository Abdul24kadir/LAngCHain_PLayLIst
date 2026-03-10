from langchain_community.document_loaders import DirectoryLoader,UnstructuredFileLoader


loader = DirectoryLoader(
    path=r"C:\1SKILL COMBACK\Langchain\6.documentloaders",
    glob="**/*",
    loader_cls=UnstructuredFileLoader
)
docs = loader.load()
print(docs[0].page_content)