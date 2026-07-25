#!/usr/bin/env python
"""
Interactive RAG Query Interface with BM25 Retrieval
Stelle Fragen zur Kubernetes-Dokumentation - beantwortet mit BM25 + Ollama Mistral
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import KUBERNETES_DATA_DIR, QUERIES_DIR, TOP_K_RETRIEVAL
from src.data.loader import load_documents
from src.rag.llm_integration import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.utils.logger import logger


def initialize_system():
    """Initialize RAG system with choice of BM25 or Embedding Retrieval"""
    
    logger.info("=" * 80)
    logger.info("RAG SYSTEM INITIALIZATION")
    logger.info("=" * 80)
    
    # 1. Check Ollama connection
    logger.info("\n[1/5] Checking Ollama connection...")
    llm = OllamaLLM()
    
    if not llm.check_connection():
        logger.error("\n❌ Ollama is not running!")
        logger.error("\nPlease start Ollama in another terminal:")
        logger.error("  ollama serve")
        logger.error("\nOr download it from: https://ollama.ai")
        return None, None, None, None
    
    logger.info("✅ Connected to Ollama")
    
    # 2. Load documents
    logger.info("\n[2/5] Loading Kubernetes documents...")
    documents = load_documents(KUBERNETES_DATA_DIR)
    
    if not documents:
        logger.error(f"\n❌ No documents found in {KUBERNETES_DATA_DIR}")
        logger.error("Please add Kubernetes YAML files to: data/kubernetes/")
        return None, None, None, None
    
    logger.info(f"✅ Loaded {len(documents)} documents")
    
    # 3. Initialize BM25 Retriever
    logger.info("\n[3/5] Initializing BM25 Retriever...")
    try:
        bm25_retriever = BM25Retriever(documents)
        logger.info("✅ BM25 Retriever initialized")
    except Exception as e:
        logger.error(f"\n❌ Failed to initialize BM25: {e}")
        return None, None, None, None
    
    # 4. Initialize Embedding Retriever
    logger.info("\n[4/5] Initializing Embedding Retriever (may take a moment)...")
    embedding_retriever = None
    try:
        embedding_retriever = EmbeddingRetriever(documents)
        logger.info("✅ Embedding Retriever initialized")
    except Exception as e:
        logger.warning(f"\n⚠️  Embedding Retriever initialization failed: {e}")
        logger.warning("   You can still use BM25 retrieval")
        embedding_retriever = None
    
    # 5. Choose retriever
    logger.info("\n[5/5] Selecting retrieval method...")
    logger.info("\n  Available retrievers:")
    logger.info("    [1] BM25 (Lexical Search) - Fast")
    if embedding_retriever:
        logger.info("    [2] Embedding (Semantic Search) - More accurate")
    
    retriever_choice = None
    while retriever_choice is None:
        try:
            choice = input("\n  Choose retriever [1/2]: ").strip()
            if choice == "1":
                retriever = bm25_retriever
                retriever_choice = "BM25"
                logger.info("  Selected: BM25 Retriever ✅")
                break
            elif choice == "2" and embedding_retriever:
                retriever = embedding_retriever
                retriever_choice = "Embedding"
                logger.info("  Selected: Embedding Retriever ✅")
                break
            elif choice == "2" and not embedding_retriever:
                logger.warning("  Embedding Retriever not available, using BM25")
                retriever = bm25_retriever
                retriever_choice = "BM25"
                break
            else:
                logger.warning("  Invalid choice, please enter 1 or 2")
        except KeyboardInterrupt:
            logger.error("\n  Interrupted by user")
            return None, None, None, None
    
    # Initialize RAG Pipeline
    try:
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            top_k_retrieval=TOP_K_RETRIEVAL
        )
        logger.info(f"✅ RAG Pipeline initialized with {retriever_choice}")
    except Exception as e:
        logger.error(f"\n❌ Failed to initialize RAG Pipeline: {e}")
        return None, None, None, None
    
    return rag_pipeline, llm, bm25_retriever, embedding_retriever


def display_retrieved_docs(result: dict, max_display: int = 3):
    """Display retrieved documents"""
    
    docs = result.get("retrieved_documents", [])
    
    if not docs:
        logger.info("  (Keine Dokumente gefunden)")
        return
    
    logger.info(f"\n  📄 Retrieved Documents ({len(docs)} total):")
    for i, doc in enumerate(docs[:max_display], 1):
        filename = doc.get("filename", "Unknown")
        score = doc.get("score", 0)
        logger.info(f"    {i}. {filename} (Score: {score:.4f})")
    
    if len(docs) > max_display:
        logger.info(f"    ... and {len(docs) - max_display} more documents")


def interactive_loop(rag_pipeline, llm, bm25_retriever, embedding_retriever):
    """Interactive question-answering loop"""
    
    logger.info("\n" + "=" * 80)
    logger.info("RAG SYSTEM READY - Interactive Mode")
    logger.info("=" * 80)
    logger.info("\n💬 Type your questions about Kubernetes (type 'exit' to quit)")
    logger.info("   Or 'help' for commands")
    logger.info("-" * 80)
    
    current_retriever = rag_pipeline.retriever
    current_retriever_name = "BM25" if current_retriever == bm25_retriever else "Embedding"
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n❓ Your question [{current_retriever_name}]: ").strip()
            
            # Handle special commands
            if user_input.lower() == "exit":
                logger.info("\n👋 Goodbye!")
                break
            
            if user_input.lower() == "help":
                logger.info("\nAvailable commands:")
                logger.info("  exit          - Quit the program")
                logger.info("  switch        - Switch between BM25 and Embedding retriever")
                logger.info("  status        - Show current settings")
                logger.info("  help          - Show this help message")
                logger.info("  Or ask any question about Kubernetes")
                continue
            
            if user_input.lower() == "switch":
                if embedding_retriever is None:
                    logger.warning("Embedding Retriever not available")
                    continue
                
                if current_retriever == bm25_retriever:
                    current_retriever = embedding_retriever
                    current_retriever_name = "Embedding"
                    rag_pipeline.retriever = embedding_retriever
                    logger.info("✅ Switched to: Embedding Retriever (Semantic Search)")
                else:
                    current_retriever = bm25_retriever
                    current_retriever_name = "BM25"
                    rag_pipeline.retriever = bm25_retriever
                    logger.info("✅ Switched to: BM25 Retriever (Lexical Search)")
                continue
            
            if user_input.lower() == "status":
                logger.info(f"\n📊 Current Status:")
                logger.info(f"   Active Retriever: {current_retriever_name}")
                logger.info(f"   Top-K Documents: {rag_pipeline.top_k_retrieval}")
                if embedding_retriever:
                    logger.info(f"   Available Retrievers: BM25, Embedding")
                else:
                    logger.info(f"   Available Retrievers: BM25 only")
                continue
            
            if not user_input:
                logger.warning("Please enter a question")
                continue
            
            # Process query through RAG pipeline
            logger.info(f"\n🔍 Processing with {current_retriever_name}: '{user_input[:60]}{'...' if len(user_input) > 60 else ''}'\n")
            
            result = rag_pipeline.run(user_input)
            
            # Display retrieved documents
            display_retrieved_docs(result)
            
            # Display answer
            answer = result.get("answer", "No answer generated")
            logger.info(f"\n🤖 Answer:\n")
            logger.info(f"   {answer}")
            
        except KeyboardInterrupt:
            logger.info("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            logger.error(f"\n❌ Error processing query: {e}")
            logger.error("   Please try again with a different question")


def main():
    """Main entry point"""
    
    # Initialize system
    rag_pipeline, llm, bm25_retriever, embedding_retriever = initialize_system()
    
    if rag_pipeline is None:
        sys.exit(1)
    
    # Start interactive loop
    interactive_loop(rag_pipeline, llm, bm25_retriever, embedding_retriever)


if __name__ == "__main__":
    main()
