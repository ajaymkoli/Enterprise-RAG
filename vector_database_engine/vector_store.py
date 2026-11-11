import os
import chromadb
from chromadb.utils import embedding_functions
from pdf_processor import process_pdf

# Save the database inside your folder, no matter where you run the script from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# The free HuggingFace model that turns text into numbers
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create (or reopen) the database on your laptop
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_fn,
)


def add_pdf_to_database(pdf_path: str) -> int:
    """Read a PDF, chunk it, embed the chunks, and store them."""
    chunks = process_pdf(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids)
    return len(chunks)


def get_relevant_chunks(query: str) -> list:
    """Take a question and return the top 3 most relevant chunks."""
    results = collection.query(query_texts=[query], n_results=3)
    return results["documents"][0]


# Quick test
if __name__ == "__main__":
    sample_path = os.path.join(BASE_DIR, "sample.pdf")
    count = add_pdf_to_database(sample_path)
    print(f"Stored {count} chunks.")

    question = "How many days of sick leave do I get?"
    for i, chunk in enumerate(get_relevant_chunks(question), start=1):
        print(f"\n--- Result {i} ---\n{chunk}")