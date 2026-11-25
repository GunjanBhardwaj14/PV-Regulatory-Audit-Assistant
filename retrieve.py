from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

query = "What are the minimum criteria for a valid adverse reaction report?"
results = vector_db.similarity_search(query, k=5)

for i, doc in enumerate(results, start=1):
    page_num = doc.metadata.get("page", 0) + 1
    print(f"\n--- MATCH {i} (PDF Page {page_num}) ---")
    print(doc.page_content)