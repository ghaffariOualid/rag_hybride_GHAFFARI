import os
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import CrossEncoder
from uuid import uuid4
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv

# Import from the existing script
from rag_chatbot_qdrant import (
    CustomEmbedding,
    CustomOpenRouterLLM,
    FullRAGPipeline,
    RAG_CONFIG,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RERANKER_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)

load_dotenv()

# Define request/response models
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

# Global variables
pipeline = None

# Custom Pipeline Class to return sources
class ApiRAGPipeline(FullRAGPipeline):
    # Safe Hybrid Retrieval override
    def _hybrid_retrieve(self, queries: List[str]) -> List[Document]:
        # Copied logic with safety check for sparse_retriever
        dense_results = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config['hybrid_top_k']}
        ).invoke(queries[0])
        
        if not self.sparse_retriever:
            return dense_results
            
        sparse_results = self.sparse_retriever.invoke(queries[0])
        
        scores = {}
        for i, doc in enumerate(dense_results):
            doc_id = hash((doc.page_content, doc.metadata.get('page', 0))) 
            scores[doc_id] = {'doc': doc, 'score': (1.0 / (i + 1 + 60))}

        for i, doc in enumerate(sparse_results):
            doc_id = hash((doc.page_content, doc.metadata.get('page', 0)))
            if doc_id in scores:
                 scores[doc_id]['score'] += (1.0 / (i + 1 + 60))
            else:
                 scores[doc_id] = {'doc': doc, 'score': (1.0 / (i + 1 + 60))}

        from operator import itemgetter
        all_candidates = sorted(scores.values(), key=itemgetter('score'), reverse=True)
        return [c['doc'] for c in all_candidates[:self.config['hybrid_top_k']]]

    def run_pipeline_with_sources(self, query: str):
        # 1. Query Expansion
        expanded_queries = self._expand_query(query)
        
        # 2. Hybrid Retrieval
        candidates = self._hybrid_retrieve(expanded_queries)
        
        # 3. BGE Re-ranking
        reranked_chunks = self._rerank(query, candidates)
        
        # 4. Cross-Encoder Verification
        verified_chunks = self._verify_and_filter(query, reranked_chunks)
        
        # Collect sources from verified_chunks
        # Returning filename as requested: ["doc1.pdf"]
        # Assuming metadata['source'] contains the full path, we extract the filename
        sources = []
        for doc in verified_chunks:
            source_path = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source_path)
            sources.append(filename)
            
        # Deduplicate sources preserving order
        unique_sources = []
        for s in sources:
            if s not in unique_sources:
                unique_sources.append(s)

        # 5. Context Assembly
        context = self._assemble_context(verified_chunks)
        
        if not context.strip():
            return "Désolé, aucune information pertinente n'a pu être trouvée dans le document.", []
        
        # Generation
        SYSTEM_PROMPT = (
            "Vous êtes un assistant IA spécialisé dans l'analyse de documents. "
            "Répondez à la question en utilisant UNIQUEMENT le contexte fourni ci-dessous. "
            "Si la réponse n'est pas dans le contexte, dites poliment que l'information n'est pas disponible dans le document. "
            "\n\nCONTEXTE: \n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT.format(context=context)),
                ("human", "{question}"),
            ]
        )
        
        try:
             answer = self.llm.invoke(prompt.format_messages(question=query)).content
        except Exception as e:
             answer = "Désolé, une erreur est survenue lors de la génération de la réponse."
             print(f"LLM Error: {e}")

        return answer, unique_sources

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    
    # Initialize models
    print("Initializing models...")
    embedder = CustomEmbedding(EMBEDDING_MODEL_NAME)
    llm = CustomOpenRouterLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device='cpu')
    
    # Connect to persistent Qdrant
    # Path must match what is used in ingest.py
    qdrant_path = os.getenv("QDRANT_PATH", "/app/qdrant_data")
    print(f"Connecting to Qdrant at: {qdrant_path}")
    
    client = QdrantClient(path=qdrant_path)
    
    # Check if collection exists
    # Check if collection exists
    if not client.collection_exists(COLLECTION_NAME):
        print(f"WARNING: Collection '{COLLECTION_NAME}' not found. Please run ingestion first using './docker.sh ingest'")
        print("Pipeline initialization skipped.")
    else:
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedder,
        )
        
        # Sparse Retriever (BM25)
        # BM25 must be rebuilt from documents usually. 
        # Since we don't carry the raw documents easily in Qdrant (payloads yes, but all of them?)
        # For a perfect RAG per requirements, we'd need to load the docs again to build BM25 
        # OR we can just use the VectorStore if BM25 is too heavy to rebuild on every startup without persisting it.
        # The requirement says "run ingestion with ./docker.sh ingest".
        # Let's try to reload the PDF just for BM25 construction if it exists, otherwise skip BM25 or handle it gracefully.
        
        PDF_PATH = os.getenv("PDF_PATH", "Schatzinsel_E.pdf")
        if os.path.exists(PDF_PATH):
            print(f"Loading PDF for BM25: {PDF_PATH}")
            loader = PyPDFLoader(PDF_PATH)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = text_splitter.split_documents(data)
            sparse_retriever = BM25Retriever.from_documents(chunks)
        else:
            print("PDF not found for BM25. Hybrid retrieval might be degraded.")
            # Fallback: create an empty BM25 or mock it? 
            # Ideally we should verify if we can skip it. The pipeline expects it.
            # We can attempt to create a dummy if needed, but easier to just check file existence.
            # Inside Docker, PDF should be mounted.
            sparse_retriever = None 

        pipeline = ApiRAGPipeline(
            vectorstore=vectorstore, 
            sparse_retriever=sparse_retriever, 
            reranker_model=reranker_model, 
            llm=llm, 
            config=RAG_CONFIG
        )
        print("Pipeline initialized.")

    yield
    # Cleanup

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized or PDF not found.")
    
    answer, sources = pipeline.run_pipeline_with_sources(request.question)
    return AskResponse(answer=answer, sources=sources)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
