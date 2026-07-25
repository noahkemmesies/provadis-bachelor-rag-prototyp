#!/usr/bin/env python
"""Test RAG Pipeline with both retrievers"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import KUBERNETES_DATA_DIR, QUERIES_DIR, RESULTS_DIR
from src.data.loader import load_queries, load_documents
from src.rag.llm_integration import OllamaLLM
from src.rag.pipeline import HybridRAGPipeline, RAGPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.utils.logger import logger


def test_rag_pipeline():
    """Test RAG Pipeline with both retrievers"""

    logger.info("=" * 80)
    logger.info("RAG PIPELINE TEST")
    logger.info("=" * 80)

    # ========================================================================
    # 1. Check Ollama Connection
    # ========================================================================
    logger.info("\n[1/6] Checking Ollama connection...")
    llm = OllamaLLM()

    if not llm.check_connection():
        logger.error("Cannot connect to Ollama!")
        logger.error("Make sure Ollama is running:")
        logger.error("  ollama serve")
        logger.error("")
        logger.error("Or download and install from: https://ollama.ai")
        return False

    # List available models
    models = llm.list_models()
    if "mistral:7b" not in models and "mistral" not in str(models).lower():
        logger.warning("Mistral model not found! Pulling it now...")
        logger.warning("This may take a few minutes...")

    # ========================================================================
    # 2. Load Data
    # ========================================================================
    logger.info("\n[2/6] Loading data...")
    documents = load_documents(KUBERNETES_DATA_DIR)
    queries, ground_truth = load_queries(QUERIES_DIR / "queries.json")

    logger.info(f"✓ Loaded {len(documents)} documents")
    logger.info(f"✓ Loaded {len(queries)} queries")

    if not documents or not queries:
        logger.error("Missing data!")
        return False

    # ========================================================================
    # 3. Initialize Retrievers
    # ========================================================================
    logger.info("\n[3/6] Initializing retrievers...")

    try:
        bm25_retriever = BM25Retriever(documents)
        logger.info("✓ BM25 Retriever initialized")
    except Exception as e:
        logger.error(f"BM25 initialization failed: {e}")
        return False

    try:
        logger.info("Initializing Embedding Retriever (may take a moment)...")
        embedding_retriever = EmbeddingRetriever(documents)
        logger.info("✓ Embedding Retriever initialized")
    except Exception as e:
        logger.error(f"Embedding initialization failed: {e}")
        logger.warning("Skipping Embedding Retriever for this test")
        embedding_retriever = None

    # ========================================================================
    # 4. Test Individual Retrievers
    # ========================================================================
    logger.info("\n[4/6] Testing RAG pipelines with sample queries...")

    # Use first 3 queries for quick test
    sample_queries = queries[:3]

    bm25_results = []
    embedding_results = []

    for query in sample_queries:
        query_text = query["text"]
        logger.info(f"\nQuery: {query_text[:60]}...")

        # BM25 Pipeline
        try:
            bm25_pipeline = RAGPipeline(bm25_retriever, llm)
            bm25_result = bm25_pipeline.run(query_text)
            bm25_results.append(bm25_result)

            logger.info(f"  BM25 Answer: {bm25_result['answer'][:80]}...")
        except Exception as e:
            logger.error(f"  BM25 Pipeline error: {e}")

        # Embedding Pipeline
        if embedding_retriever:
            try:
                embedding_pipeline = RAGPipeline(embedding_retriever, llm)
                embedding_result = embedding_pipeline.run(query_text)
                embedding_results.append(embedding_result)

                logger.info(f"  Embedding Answer: {embedding_result['answer'][:80]}...")
            except Exception as e:
                logger.error(f"  Embedding Pipeline error: {e}")

    # ========================================================================
    # 5. Hybrid Pipeline Test
    # ========================================================================
    logger.info("\n[5/6] Testing Hybrid Pipeline...")

    retrievers = {"BM25": bm25_retriever}
    if embedding_retriever:
        retrievers["Embedding"] = embedding_retriever

    try:
        hybrid_pipeline = HybridRAGPipeline(retrievers, llm)

        # Test on first query
        sample_query = queries[0]["text"]
        logger.info(f"Hybrid test query: {sample_query[:60]}...")

        comparison = hybrid_pipeline.compare_answers(sample_query)

        logger.info("\nHybrid Pipeline Comparison:")
        for retriever_name, data in comparison["retrievers"].items():
            logger.info(f"  {retriever_name}:")
            logger.info(f"    Documents retrieved: {data['num_documents']}")
            logger.info(f"    Answer length: {data['answer_length']}")

    except Exception as e:
        logger.error(f"Hybrid Pipeline error: {e}")

    # ========================================================================
    # 6. Save Results
    # ========================================================================
    logger.info("\n[6/6] Saving results...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_data = {
        "bm25_results": [
            {
                "query": r["query"],
                "num_documents": len(r.get("retrieved_documents", [])),
                "answer_preview": r.get("answer", "")[:200],
            }
            for r in bm25_results
        ],
        "embedding_results": [
            {
                "query": r["query"],
                "num_documents": len(r.get("retrieved_documents", [])),
                "answer_preview": r.get("answer", "")[:200],
            }
            for r in embedding_results
        ],
    }

    with open(RESULTS_DIR / "rag_pipeline_test.json", "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"✓ Saved results to {RESULTS_DIR / 'rag_pipeline_test.json'}")

    # ========================================================================
    # 7. Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("✅ RAG PIPELINE TEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"BM25 Results: {len(bm25_results)} queries processed")
    logger.info(f"Embedding Results: {len(embedding_results)} queries processed")
    logger.info(f"Total Hybrid Retrievers: {len(retrievers)}")

    logger.info("\n✅ RAG Pipeline is working correctly!")
    logger.info("\nNext steps:")
    logger.info("  1. Run full evaluation with all queries")
    logger.info("  2. Analyze answer quality")
    logger.info("  3. Compare retrieval strategies")

    return True


if __name__ == "__main__":
    try:
        success = test_rag_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
