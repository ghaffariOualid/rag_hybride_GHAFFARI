
import os
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from operator import itemgetter
import random 

from openai import OpenAI

# LangChain Imports
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4

# AI/ML Models
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Configuration RAGio ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Pipeline Configs (Fidèle à RAGio)
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200 
COLLECTION_NAME = "ragio_pdf_collection"
RAG_CONFIG = {
    "num_expansions": 4,          # Étape 1: Query Expansion
    "hybrid_top_k": 50,           # Étape 2: Candidats avant re-classement
    "rerank_top_k": 20,           # Étape 3: Nombre après re-classement
    "verify_threshold": 0.5,      # Étape 4: Seuil de vérification (valeur arbitraire pour la démo)
}


if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY non défini. Veuillez le configurer dans le fichier .env.")

# --- CLASSES DE BASE FIDÈLES À RAGIO ---

class CustomEmbedding(Embeddings):
    """Implémentation fidèle de l'Embedding de RAGio."""
    def __init__(self, model_name: str):
        if isinstance(model_name, str):
            self.model_instance = SentenceTransformer(model_name)
        else:
            self.model_instance = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model_instance.encode(text).tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model_instance.encode(text).tolist()

class CustomOpenRouterLLM(Runnable):
    """Wrapper pour OpenRouter avec logique de fallback simplifiée."""
    def __init__(self, model, temperature=0.6, fallback_models=None):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self.model = model
        self.temperature = temperature
        self.fallback_models = fallback_models or [] 
    
    def _format_messages(self, prompt):
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [{"role": "user", "content": str(prompt)}]
    
    def invoke(self, prompt, config=None):
        formatted_messages = self._format_messages(prompt)
        models_to_try = [self.model] + self.fallback_models
        
        for model in models_to_try:
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    temperature=self.temperature
                )
                return AIMessage(content=completion.choices[0].message.content)
            except Exception:
                continue 
        
        raise Exception("Erreur LLM: Tous les modèles OpenRouter ont échoué.")


# --- PIPELINE RAG 5 ÉTAPES FIDELE ---

class FullRAGPipeline:
    """Orchestre les 5 étapes du pipeline RAGio pour une fidélité totale."""
    
    def __init__(self, vectorstore, sparse_retriever, reranker_model, llm, config):
        self.vectorstore = vectorstore
        self.sparse_retriever = sparse_retriever
        self.reranker_model = reranker_model
        self.llm = llm
        self.config = config

    # --- Étape 1: Query Expansion (Expansion de Requête) ---
    def _expand_query(self, query: str) -> List[str]:
        """Génère des reformulations de la requête en utilisant le LLM."""
        prompt_template = """
        Vous êtes un expert en recherche. Générez {num} requêtes de recherche alternatives et pertinentes pour la question suivante.
        Séparez chaque requête par un point-virgule (;).
        Question: {query}
        Requêtes (inclure l'original):
        """
        try:
            expanded_prompt = prompt_template.format(
                num=self.config['num_expansions'],
                query=query
            )
            # Utilise l'invoke LLM pour générer les expansions
            result = self.llm.invoke(expanded_prompt).content
            
            queries = [q.strip() for q in result.split(';') if q.strip()]
            if query not in queries:
                 queries.append(query)
            return queries[:self.config['num_expansions']]
        except Exception:
            # En cas d'échec de l'expansion, retourne la requête originale (fail-safe)
            return [query] 

    # --- Étape 2: Hybrid Retrieval (Récupération Hybride) ---
    def _hybrid_retrieve(self, queries: List[str]) -> List[Document]:
        """Combine Dense (Qdrant) et Sparse (BM25) avec fusion simulée (RRF-like)."""
        
        # Récupération Dense sur la requête principale
        dense_results: List[Document] = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config['hybrid_top_k']}
        ).invoke(queries[0])
        
        # Récupération Sparse/BM25
        sparse_results: List[Document] = self.sparse_retriever.invoke(queries[0])
        
        # Fusion des rangs (Simule la Reciprocal Rank Fusion)
        scores = {}
        # Combinaison des résultats et attribution de scores
        for i, doc in enumerate(dense_results):
            doc_id = hash((doc.page_content, doc.metadata.get('page', 0))) 
            scores[doc_id] = {'doc': doc, 'score': (1.0 / (i + 1 + 60))}

        for i, doc in enumerate(sparse_results):
            doc_id = hash((doc.page_content, doc.metadata.get('page', 0)))
            # Ajoute le score BM25 au score dense existant (fusion)
            if doc_id in scores:
                 scores[doc_id]['score'] += (1.0 / (i + 1 + 60))
            else:
                 scores[doc_id] = {'doc': doc, 'score': (1.0 / (i + 1 + 60))}

        # Trie par score et retourne le top_k hybride
        all_candidates = sorted(scores.values(), key=itemgetter('score'), reverse=True)
        return [c['doc'] for c in all_candidates[:self.config['hybrid_top_k']]]

    # --- Étape 3: BGE Re-ranking (Re-classement) ---
    def _rerank(self, query: str, chunks: List[Document]) -> List[Document]:
        """Applique le re-classement Cross-Encoder BGE."""
        if not self.reranker_model or not chunks:
            return chunks

        pairs = [(query, doc.page_content) for doc in chunks]
        scores = self.reranker_model.predict(pairs).tolist()
        
        scored_docs = sorted(zip(chunks, scores), key=itemgetter(1), reverse=True)
        
        # Retourne le top_k après re-classement
        return [doc for doc, score in scored_docs[:self.config['rerank_top_k']]]

    # --- Étape 4: Cross-Encoder Verification (Vérification) ---
    def _verify_and_filter(self, query: str, chunks: List[Document]) -> List[Document]:
        """Filtre les chunks dont le score est inférieur au seuil de vérification."""
        if not self.reranker_model or not chunks:
            return chunks
        
        # Le modèle est ré-exécuté car les scores Cross-Encoder ne sont pas passés entre les étapes
        pairs = [(query, doc.page_content) for doc in chunks]
        scores = self.reranker_model.predict(pairs).tolist()
        
        verified_docs = []
        for doc, score in zip(chunks, scores):
            # Le seuil est appliqué ici pour simuler l'étape de vérification explicite
            if score >= self.config['verify_threshold']:
                verified_docs.append(doc)
        
        return verified_docs

    # --- Étape 5: Context Assembly (Assemblage de Contexte) ---
    def _assemble_context(self, chunks: List[Document]) -> str:
        """Formate et assemble le contexte pour le LLM."""
        # Note: Les fonctions de déduplication et de gestion du budget de tokens sont omises
        return "\n\n".join(
            f"--- Source (Page {doc.metadata.get('page', 'N/A')}) ---\n{doc.page_content}" 
            for doc in chunks
        )

    # --- Méthode Principale d'Exécution ---
    def run_pipeline(self, query: str) -> str:
        """Exécute l'intégralité du pipeline 5 étapes."""
        
        # 1. Query Expansion 
        print(f"Étape 1: Expansion de Requête...")
        expanded_queries = self._expand_query(query)
        print(f"   -> Requêtes générées: {expanded_queries}")

        # 2. Hybrid Retrieval
        print("Étape 2: Récupération Hybride (Dense + Sparse/BM25)...")
        candidates = self._hybrid_retrieve(expanded_queries)
        print(f"   -> Candidats récupérés: {len(candidates)}")

        # 3. BGE Re-ranking
        print("Étape 3: Re-classement BGE...")
        reranked_chunks = self._rerank(query, candidates)
        print(f"   -> Chunks après Re-classement: {len(reranked_chunks)}")

        # 4. Cross-Encoder Verification
        print("Étape 4: Vérification Cross-Encoder (Seuil)...")
        verified_chunks = self._verify_and_filter(query, reranked_chunks)
        print(f"   -> Chunks vérifiés: {len(verified_chunks)}")

        # 5. Context Assembly
        print("Étape 5: Assemblage de Contexte...")
        context = self._assemble_context(verified_chunks)
        
        if not context.strip():
            return "Désolé, aucune information pertinente n'a pu être trouvée dans le document après vérification par le pipeline RAG."
        
        # Génération finale
        SYSTEM_PROMPT = (
            "Vous êtes un assistant IA spécialisé dans l'analyse de documents. "
            "Répondez à la question en utilisant UNIQUEMENT le contexte fourni ci-dessous, qui provient du document 'Schatzinsel_E.pdf'. "
            "Indiquez toujours la page du document source lorsque vous fournissez une réponse, par exemple : [Page X]. "
            "Si la réponse n'est pas dans le contexte, dites poliment que l'information n'est pas disponible dans le document. "
            "\n\nCONTEXTE: \n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT.format(context=context)),
                ("human", "{question}"),
            ]
        )
        
        return self.llm.invoke(prompt.format_messages(question=query)).content
    
# --- 3. Initialisation et Lancement ---

if __name__ == "__main__":
    
    # Initialisation des modèles et des clients
    embedder = CustomEmbedding(EMBEDDING_MODEL_NAME)
    llm = CustomOpenRouterLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device='cpu')

    PDF_PATH = "Schatzinsel_E.pdf"

    try:
        # Chargement et Chunking 
        print(f"Chargement et découpage de: {PDF_PATH} (Taille: {CHUNK_SIZE}, Chevauchement: {CHUNK_OVERLAP})...")
        loader = PyPDFLoader(PDF_PATH)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(data)
        
        # Indexation Qdrant (Dense) et BM25 (Sparse)
        print(f"Indexation de {len(chunks)} fragments dans Qdrant (Dense) et BM25 (Sparse)...")

        # Dense Retriever (Qdrant avec stockage local)
        # Create Qdrant client and collection explicitly to avoid network lookups
        qdrant_path = "/tmp/qdrant_storage"
        client = QdrantClient(path=qdrant_path)

        # Determine vector size from the SentenceTransformer model
        try:
            vector_size = embedder.model_instance.get_sentence_embedding_dimension()
        except Exception:
            # Fallback: try attribute name used by some models
            vector_size = getattr(embedder.model_instance, "get_sentence_embedding_dimension", lambda: None)()

        # Create collection if it doesn't exist
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        except Exception:
            # If collection exists, ignore
            pass

        # Build the vectorstore and add documents with uuids
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedder,
        )

        uuids = [str(uuid4()) for _ in range(len(chunks))]
        vectorstore.add_documents(documents=chunks, ids=uuids)
        
        # Sparse Retriever (BM25 in-memory)
        sparse_retriever = BM25Retriever.from_documents(chunks)
        
        # Construction du Pipeline
        pipeline = FullRAGPipeline(
            vectorstore=vectorstore, sparse_retriever=sparse_retriever, reranker_model=reranker_model, llm=llm, config=RAG_CONFIG
        )
        
    except Exception as e:
        print(f"\n[ERREUR D'INITIALISATION FATALE] : {e}")
        exit()

    # Boucle d'Interaction
    print("\n" + "="*70)
    print("🤖 Chatbot RAGio (Pipeline 5 Étapes - Fidélité Totale) PRÊT.")
    print("   Tapez 'quitter' pour arrêter le chat.")
    print("="*70)

    while True:
        try:
            question = input("👤 Vous: ")
            if question.lower() in ['quitter', 'exit', 'quit']:
                print("Aurevoir!")
                break
            
            if not question.strip():
                continue

            print("\n--- Début de l'Exécution du Pipeline ---")
            
            # Exécution de l'intégralité du pipeline 5 étapes
            answer = pipeline.run_pipeline(question)
            
            print("--- Fin de l'Exécution du Pipeline ---")
            print(f"\n🤖 RAGio: {answer}")
            print("\n" + "-"*70)
            
        except Exception as e:
            print(f"\n[ERREUR D'EXÉCUTION] : {e}")
            break