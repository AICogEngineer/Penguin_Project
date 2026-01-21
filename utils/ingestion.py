
import os
import shutil
from dotenv import load_dotenv
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_docling import DoclingLoader

load_dotenv()

#Use Markdown header text splitting to get more proper precise data and chunks

PDF_PATH = os.getenv("POLICY_PDF_PATH")
DB_PATH = os.getenv("CHROMA_DB_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
pipeline_options = PdfPipelineOptions()
pipeline_options.allow_external_plugins = True


if not PDF_PATH or not DB_PATH or not EMBEDDING_MODEL:
    raise ValueError("POLICY_PDF_PATH, CHROMA_DB_PATH, and EMBEDDING_MODEL must be set in the environment.")

def ingest_policy():
    print(f"Loading policy from {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Policy file not found at {PDF_PATH}")

    loader = DoclingLoader(PDF_PATH, chunker=HybridChunker(tokenizer=EMBEDDING_MODEL))
    docs = loader.load()
    splits = docs
    
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
