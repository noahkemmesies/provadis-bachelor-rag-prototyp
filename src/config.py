"""
Zentrale Konfiguration für RAG Kubernetes Retrieval Comparison
"""

import os
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KUBERNETES_DATA_DIR = DATA_DIR / "kubernetes"
QUERIES_DIR = DATA_DIR / "queries"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, KUBERNETES_DATA_DIR, QUERIES_DIR, PROCESSED_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Retrieval Settings
# ============================================================================

# BM25 Configuration
BM25_LANGUAGE = "english"
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Length normalization parameter

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence-Transformers Model
EMBEDDING_DIMENSION = 384  # MiniLM output dimension
EMBEDDING_DEVICE = "cpu"  # "cpu" oder "cuda"

# Vector Database Configuration
VECTOR_DB_TYPE = "chromadb"  # "chromadb" oder "faiss"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_data"

# ============================================================================
# Retrieval Common
# ============================================================================

TOP_K_RETRIEVAL = 5  # Number of documents to retrieve
RETRIEVER_TIMEOUT = 30  # seconds

# ============================================================================
# RAG Pipeline Settings
# ============================================================================

# LLM Configuration
LLM_PROVIDER = "ollama"  # Currently only ollama supported
LLM_MODEL = "mistral:7b"
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 1000
LLM_TIMEOUT = 60  # seconds

# Ollama Settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_RETRIES = 3

# Context Settings
CONTEXT_WINDOW = 5  # Number of documents to include as context
CONTEXT_MAX_CHARS = 4000  # Max characters for context

# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are a helpful assistant answering questions about Kubernetes.
Use the provided context to answer questions accurately and concisely.
If the context doesn't contain relevant information, say so."""

QUESTION_PROMPT = """Context:
{context}

Question: {question}

Answer:"""

# ============================================================================
# Evaluation Settings
# ============================================================================

# Evaluation Metrics to compute
EVALUATION_METRICS = [
    "precision@5",
    "precision@10",
    "recall@5",
    "recall@10",
    "f1@5",
    "f1@10",
    "mrr",
    "ndcg@5",
    "ndcg@10",
]

# Ground Truth Settings
GROUND_TRUTH_FILE = QUERIES_DIR / "ground_truth.json"
QUERIES_FILE = QUERIES_DIR / "queries.json"

# ============================================================================
# Data Processing Settings
# ============================================================================

# Text Preprocessing
LOWERCASE = True
REMOVE_PUNCTUATION = True
REMOVE_EXTRA_WHITESPACE = True
MIN_TOKEN_LENGTH = 2

# YAML Parsing
YAML_ENCODING = "utf-8"

# ============================================================================
# Logging Settings
# ============================================================================

LOG_LEVEL = "INFO"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rag_evaluation.log"

# ============================================================================
# Cache Settings
# ============================================================================

CACHE_EMBEDDINGS = True
EMBEDDINGS_CACHE_FILE = PROCESSED_DIR / "embeddings.pkl"
DOCUMENTS_CACHE_FILE = PROCESSED_DIR / "documents.pkl"

# ============================================================================
# Performance Settings
# ============================================================================

BATCH_SIZE = 32  # Batch size for embedding computation
NUM_WORKERS = 4  # Number of workers for data loading

# ============================================================================
# Debug Settings
# ============================================================================

DEBUG = False
VERBOSE = False
