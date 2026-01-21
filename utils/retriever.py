
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not PINECONE_INDEX_NAME or not EMBEDDING_MODEL:
    raise ValueError("PINECONE_INDEX_NAME and EMBEDDING_MODEL must be set in the environment.")

import threading

# Global cache for the embedding model to prevent re-loading weights on every call
_cached_embeddings = None
_init_lock = threading.Lock()

# fetch embeddings
def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        with _init_lock:
            # Double check locking pattern
            if _cached_embeddings is None:
                print("Loading Embedding Model (Active Singleton)...")
                _cached_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _cached_embeddings

def get_policy_retriever(k=4):
    """
    Returns the VectorStoreRetriever for the policy document.
    
    k =  num of chunks to retrieve **SET TO FOUR RIGHT NOW BUT CAN BE CHANGED**
    
    Returns VectorStoreRetriever -> langchain object that can .invoke(query) to find docs.
    """
    embeddings = get_embeddings()
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": k})
