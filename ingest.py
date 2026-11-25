import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

db_path = "./chroma_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

pdf_path = "./pv_data/ema_gvp_module_vi.pdf"

loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(pages)
print(f"Successfully sliced the document into {len(chunks)} chunks.")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=db_path
)

print("Successfully built and saved the Vector Vault in ./chroma_db!")