import chromadb
import time
from typing import List
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path
from langchain_core.messages import BaseMessage,SystemMessage

class VectorStore:
    def __init__(self, user_id:str):
        self.user_id = user_id

        path = Path(f"storage/{user_id}/chroma_db")
        path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(path))

        self.embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        self.collection = self.client.get_or_create_collection(
            name='chat_history',
            embedding_function=self.embed_fn # type: ignore
        )

    def index(self, messages: List[BaseMessage], session_id: str):
        ids = []
        documents = []
        metadatas = []

        for i, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                continue
            ids.append(f"{session_id}_{i}")
            documents.append(str(message.content))
            metadatas.append({
                "role": message.type,
                "session_id": session_id,
                "user_id": self.user_id,
                "timestamp": time.time()
            })
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        return len(ids)
