
import os
import shutil
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter

load_dotenv()

PDF_PATH = os.getenv("POLICY_PDF_PATH")
DB_PATH = os.getenv("CHROMA_DB_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not PDF_PATH or not DB_PATH or not EMBEDDING_MODEL:
    raise ValueError("POLICY_PDF_PATH, CHROMA_DB_PATH, and EMBEDDING_MODEL must be set in the environment.")

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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split policy into {len(splits)} chunks.")

    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating/Updating Vector Store at {DB_PATH}...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_policy()
