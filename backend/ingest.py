import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4
from dotenv import load_dotenv

# Reuse config from existing file
from rag_chatbot_qdrant import (
    CustomEmbedding,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)

load_dotenv()

# Data Persistence Path
QDRANT_PATH = os.getenv("QDRANT_PATH", "/app/qdrant_data")
PDF_PATH = os.getenv("PDF_PATH", "Schatzinsel_E.pdf")

def ingest():
    print("--- Starting Ingestion ---")
    
    # 1. Initialize Embeddings
    print(f"Initializing Embeddings: {EMBEDDING_MODEL_NAME}")
    embedder = CustomEmbedding(EMBEDDING_MODEL_NAME)

    # 2. Load PDF
    if not os.path.exists(PDF_PATH):
        print(f"Error: File {PDF_PATH} not found.")
        return

    print(f"Loading PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()

    # 3. Split Text
    print(f"Splitting text (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(data)
    print(f"Created {len(chunks)} chunks.")

    # 4. Indexing
    print(f"Indexing to Qdrant at {QDRANT_PATH}...")
    
    client = QdrantClient(path=QDRANT_PATH)

    # Vector size determination
    try:
        vector_size = embedder.model_instance.get_sentence_embedding_dimension()
    except:
         vector_size = getattr(embedder.model_instance, "get_sentence_embedding_dimension", lambda: None)()
    
    # Re-create collection to ensure fresh start
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
    )
    
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vectorstore.add_documents(documents=chunks, ids=uuids)
    
    print("--- Ingestion Complete ---")
    print(f"Stored {len(chunks)} vectors in collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    ingest()
