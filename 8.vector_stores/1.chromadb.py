from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langchain_core.documents import Document

doc1 = Document(
    page_content="""
Mumbai Indians are one of the most successful teams in IPL history with 5 championships (2013, 2015, 2017, 2019, 2020).
The team is captained by Rohit Sharma for many years and has star players like Jasprit Bumrah, Hardik Pandya, and Suryakumar Yadav.
Mumbai Indians play their home matches at Wankhede Stadium in Mumbai.
The team is known for its strong batting lineup and world-class fast bowling attack.
""",
    metadata={"team": "Mumbai Indians", "city": "Mumbai", "titles": 5}
)

doc2 = Document(
    page_content="""
Chennai Super Kings (CSK) is one of the most consistent teams in IPL history and has won 5 IPL titles.
The team has been led by legendary captain MS Dhoni for most seasons.
Key players include Ravindra Jadeja, Ruturaj Gaikwad, and Deepak Chahar.
CSK plays its home matches at MA Chidambaram Stadium in Chennai.
The team is famous for its loyal fan base called the Yellow Army.
""",
    metadata={"team": "Chennai Super Kings", "city": "Chennai", "titles": 5}
)

doc3 = Document(
    page_content="""
Royal Challengers Bangalore (RCB) is one of the most popular IPL teams.
The team has featured star players like Virat Kohli, AB de Villiers, Chris Gayle, and Faf du Plessis.
RCB plays its home matches at M. Chinnaswamy Stadium in Bangalore.
Despite having strong batting lineups, the team has not yet won an IPL title.
The team is known for aggressive batting and passionate fan support.
""",
    metadata={"team": "Royal Challengers Bangalore", "city": "Bangalore", "titles": 0}
)

doc4 = Document(
    page_content="""
Kolkata Knight Riders (KKR) is an IPL franchise owned by Bollywood actor Shah Rukh Khan.
The team has won 2 IPL championships in 2012 and 2014 under captain Gautam Gambhir.
Key players have included Andre Russell, Sunil Narine, and Shreyas Iyer.
KKR plays its home matches at Eden Gardens in Kolkata.
The team is known for aggressive gameplay and strong all-round performances.
""",
    metadata={"team": "Kolkata Knight Riders", "city": "Kolkata", "titles": 2}
)

doc5 = Document(
    page_content="""
Sunrisers Hyderabad (SRH) is known for its strong bowling attack.
The team won the IPL championship in 2016 under captain David Warner.
Important players have included Bhuvneshwar Kumar, Kane Williamson, and Rashid Khan.
SRH plays its home matches at Rajiv Gandhi International Cricket Stadium in Hyderabad.
The team focuses on disciplined bowling and balanced team performance.
""",
    metadata={"team": "Sunrisers Hyderabad", "city": "Hyderabad", "titles": 1}
)

docs = [doc1, doc2, doc3, doc4, doc5]


embedding = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='chroma_db',
    collection_name='ipl_teams'
)
# add documents to vector store
vector_store.add_documents(docs)
print(vector_store)

#get documents 
result1 = vector_store.get(include=['embeddings','documents','metadatas'])
print(result1)
# similarity search - search relevant docs
result2 = vector_store.similarity_search(
    query="which team did Virat KOhli play for?",
    k = 2
)
for doc in result2:
    print(doc.page_content)
    print('--------')
result3 = vector_store.similarity_search_with_score(
    query = 'which team did virat kolhi play for?',
    k=2
)
for doc, score in result3:
    print(doc.page_content)
    print("Score:", score)
    print("--------")

result4 = vector_store.similarity_search_with_score(
    query='',
    filter={'team':'Kolkata Knight Riders'}
)
print(result4)
doc6 = Document(
    page_content="Gujarat Titans won IPL 2022",
    metadata={"team": "GT"}
)


vector_store.add_documents([doc6])

result5 = vector_store.get(include=['embeddings','documents','metadatas'])
print(result5)

