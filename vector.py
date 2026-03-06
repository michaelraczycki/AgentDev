from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'reviews.csv'))
embeddings = OllamaEmbeddings(model='mxbai-embed-large')

db_location = "./chrome_langchain_db"
add_documents= not os.path.exists(db_location)

if add_documents:
    documents=[]
    ids = []
    for i, row in df.iterrows():
        documents.append(Document(
            page_content=row['restaurant_name']+ " "+ row['review'],
            metadata ={"rating": row['star_rating'], 'location': row['address']},
            id = str(i)        
        ))
        ids.append(str(i))

vectorstore = Chroma(collection_name="restaurant_reviews", persist_directory=db_location, embedding_function=embeddings)
if add_documents:
    vectorstore.add_documents(documents=documents, ids=ids)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}

)