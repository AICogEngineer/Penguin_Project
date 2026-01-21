
import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_core.documents import Document

load_dotenv()

PDF_PATH = os.getenv("POLICY_PDF_PATH")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def ingest_policy_hybrid():
    print(f"Loading policy from {PDF_PATH} (HYBRID MODE)...")

    converter = DocumentConverter()
    doc_result = converter.convert(PDF_PATH)
    
    chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2") 
    chunks = chunker.chunk(doc_result.document)
    
    lc_docs = [
        Document(
            page_content=chunk.text,
            metadata={"source": PDF_PATH, **chunk.meta.export_json_dict()}
        )
        for chunk in chunks
    ]
    
    print(f"Split policy into {len(lc_docs)} hybrid chunks.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)

    print(f"Upserting to Pinecone index {PINECONE_INDEX_NAME}...")
    PineconeVectorStore.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("Hybrid Ingestion complete!")

if __name__ == "__main__":
     ingest_policy_hybrid()
