from langchain_text_splitters import CharacterTextSplitter
text = """
Artificial Intelligence (AI) is a branch of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence. These tasks include learning, reasoning, problem solving, understanding natural language, recognizing patterns, and making decisions. Over the past decade, AI has evolved rapidly due to advances in machine learning, deep learning, and the availability of large datasets.

Machine learning is a subfield of AI that enables computers to learn from data without being explicitly programmed. Instead of following fixed instructions, machine learning models analyze patterns in data and make predictions or decisions based on those patterns. Common applications of machine learning include recommendation systems, spam detection, fraud detection, and image recognition.

Deep learning is a specialized area of machine learning that uses neural networks with many layers to process complex data. These neural networks are inspired by the structure of the human brain. Deep learning has enabled breakthroughs in areas such as speech recognition, computer vision, and natural language processing. Technologies like voice assistants, self-driving cars, and automated translation systems rely heavily on deep learning models.

Natural Language Processing (NLP) is a field of AI that focuses on enabling machines to understand and generate human language. NLP combines linguistics, computer science, and machine learning to analyze text and speech. Applications of NLP include chatbots, language translation, sentiment analysis, and text summarization. Modern NLP systems often rely on large language models such as GPT and BERT, which are trained on massive amounts of text data.

One of the most important challenges in building AI systems is managing large amounts of information efficiently. This is where techniques like text splitting, embeddings, and vector databases become important. When dealing with long documents, it is often necessary to break the text into smaller chunks before processing it with machine learning models. This process is called text splitting.

Text splitting is especially important in Retrieval Augmented Generation (RAG) systems. In a RAG pipeline, documents are first loaded from sources such as PDFs, websites, or databases. The text is then split into smaller chunks using text splitters like RecursiveCharacterTextSplitter. These chunks are converted into embeddings, which are numerical representations of the text. The embeddings are stored in a vector database where they can be efficiently searched and retrieved when answering user queries.

When a user asks a question, the system retrieves the most relevant chunks from the vector database. These chunks are then passed to a language model along with the user’s question. The model uses this additional context to generate a more accurate and relevant response. This approach improves the reliability of AI systems by grounding their responses in actual documents instead of relying only on the model’s internal knowledge.

As AI continues to develop, tools like LangChain help developers build applications that combine language models, data sources, and reasoning capabilities. These tools make it easier to create chatbots, document question-answering systems, and intelligent assistants that can interact with large amounts of information.

"""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=''
)
result = splitter.split_text(text)
print(result)