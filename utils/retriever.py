import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DB_PATH = os.getenv("CHROMA_DB_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not DB_PATH or not EMBEDDING_MODEL:
    raise ValueError("CHROMA_DB_PATH and EMBEDDING_MODEL must be set in the environment.")

# Global cache for the embedding model to prevent re-loading weights on every call
_cached_embeddings = None

# fetch embeddings
def get_embeddings():
    global _cached_embeddings
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
    
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": k})
