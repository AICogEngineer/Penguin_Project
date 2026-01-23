
import os
import shutil
import time
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter

load_dotenv()

PDF_PATH = os.getenv("POLICY_PDF_PATH")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not PDF_PATH or not PINECONE_API_KEY or not PINECONE_INDEX_NAME or not EMBEDDING_MODEL:
    raise ValueError("POLICY_PDF_PATH, PINECONE_API_KEY, PINECONE_INDEX_NAME, and EMBEDDING_MODEL must be set in the environment.")

class DoclingLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.converter = DocumentConverter()

    def lazy_load(self):
        print(f"Converting {self.file_path} with Docling...")
        result = self.converter.convert(self.file_path)
        md_content = result.document.export_to_markdown()
        yield Document(page_content=md_content, metadata={"source": self.file_path})

def ingest_policy():
    print(f"Loading policy from {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Policy file not found at {PDF_PATH}")

    loader = DoclingLoader(PDF_PATH)
    docs = list(loader.lazy_load())
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_text = docs[0].page_content
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = markdown_splitter.split_text(markdown_text)
    
    # ensure metadata is preserved
    for split in splits:
        split.metadata["source"] = PDF_PATH
    print(f"Split policy into {len(splits)} chunks.")

    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Initializing Pinecone index: {PINECONE_INDEX_NAME}...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index {PINECONE_INDEX_NAME}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384, # all-MiniLM-L6-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)

    print(f"Upserting to Pinecone index {PINECONE_INDEX_NAME}...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_policy()
