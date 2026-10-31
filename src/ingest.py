
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Project paths
PDF_FOLDER = Path("data/policies")
VECTOR_DB_PATH = "./vector_store"
COLLECTION_NAME = "shopassist_policies"

# --------------------------------------------------
# STEP 1: PDF LOADING
# --------------------------------------------------

documents = []

pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF found inside data/policies/"
    )

for pdf_file in pdf_files:
    print(f"Loading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()

    documents.extend(pages)

print(f"\nTotal pages loaded: {len(documents)}")

# --------------------------------------------------
# STEP 2: CHUNKING
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# Inspect first chunk
if chunks:
    print("\nFirst chunk:")
    print(chunks[0].page_content)

# --------------------------------------------------
# STEP 3: FREE EMBEDDING MODEL
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Test one embedding
test_vector = embedding_model.embed_query(
    "What is the refund policy?"
)

print(f"\nEmbedding dimension: {len(test_vector)}")

# --------------------------------------------------
# STEP 4: CHROMADB VECTOR STORAGE
# --------------------------------------------------

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=VECTOR_DB_PATH
)

# Remove old indexed records before re-ingesting
existing_records = vector_db.get()

if existing_records["ids"]:
    vector_db.delete(ids=existing_records["ids"])

# Generate embeddings and store documents
vector_db.add_documents(documents=chunks)

print("\nSuccessfully stored chunks in ChromaDB!")
print("Total stored chunks:", vector_db._collection.count())
