#!/usr/bin/env python
"""Quick test of retriever implementations"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import KUBERNETES_DATA_DIR, QUERIES_DIR, TOP_K_RETRIEVAL
from src.data.loader import load_queries, load_documents
from src.evaluation.metrics import MetricsCalculator, precision_at_k, recall_at_k
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.utils.logger import logger


def test_retrievers():
    """Quick test of both retrievers"""

    logger.info("=" * 80)
    logger.info("QUICK RETRIEVER TEST")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading data...")
    documents = load_documents(KUBERNETES_DATA_DIR)
    queries, ground_truth = load_queries(QUERIES_DIR / "queries.json")

    logger.info(f"✓ Loaded {len(documents)} documents")
    logger.info(f"✓ Loaded {len(queries)} queries")

    if not documents or not queries:
        logger.error("Missing data!")
        return False

    # Test BM25
    logger.info("\n" + "-" * 80)
    logger.info("Testing BM25 Retriever...")
    logger.info("-" * 80)

    try:
        bm25 = BM25Retriever(documents)
        query_text = queries[0]["text"]
        query_id = queries[0]["id"]

        logger.info(f"Query: {query_text}")
        results = bm25.retrieve(query_text, top_k=TOP_K_RETRIEVAL)

        logger.info(f"✓ BM25 Retrieved {len(results)} documents:")
        for i, doc in enumerate(results[:3], 1):
            logger.info(f"  {i}. {doc['filename']} (score: {doc['score']:.4f})")

        # Check metrics
        retrieved_ids = [doc["filename"] for doc in results]
        relevant_ids = ground_truth.get(query_id, [])
        precision = precision_at_k(retrieved_ids, relevant_ids, TOP_K_RETRIEVAL)
        recall = recall_at_k(retrieved_ids, relevant_ids, TOP_K_RETRIEVAL)

        logger.info(f"  Precision@{TOP_K_RETRIEVAL}: {precision:.4f}")
        logger.info(f"  Recall@{TOP_K_RETRIEVAL}: {recall:.4f}")
        logger.info("✓ BM25 test PASSED")

    except Exception as e:
        logger.error(f"✗ BM25 test FAILED: {e}")
        return False

    # Test Embeddings
    logger.info("\n" + "-" * 80)
    logger.info("Testing Embedding Retriever...")
    logger.info("-" * 80)

    try:
        logger.info("Initializing Embedding Retriever (may take a moment)...")
        embeddings = EmbeddingRetriever(documents)
        logger.info("✓ Embedding Retriever initialized")

        query_text = queries[0]["text"]
        query_id = queries[0]["id"]

        logger.info(f"Query: {query_text}")
        results = embeddings.retrieve(query_text, top_k=TOP_K_RETRIEVAL)

        logger.info(f"✓ Embeddings Retrieved {len(results)} documents:")
        for i, doc in enumerate(results[:3], 1):
            logger.info(f"  {i}. {doc['filename']} (score: {doc['score']:.4f})")

        # Check metrics
        retrieved_ids = [doc["filename"] for doc in results]
        relevant_ids = ground_truth.get(query_id, [])
        precision = precision_at_k(retrieved_ids, relevant_ids, TOP_K_RETRIEVAL)
        recall = recall_at_k(retrieved_ids, relevant_ids, TOP_K_RETRIEVAL)

        logger.info(f"  Precision@{TOP_K_RETRIEVAL}: {precision:.4f}")
        logger.info(f"  Recall@{TOP_K_RETRIEVAL}: {recall:.4f}")
        logger.info("✓ Embedding test PASSED")

    except Exception as e:
        logger.error(f"✗ Embedding test FAILED: {e}")
        return False

    # ========================================================================
    # Comparison
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("QUICK COMPARISON")
    logger.info("=" * 80)

    metrics_calc = MetricsCalculator(k_values=[5, 10])

    # Test all queries
    logger.info(f"\nTesting all {len(queries)} queries...")

    bm25_all_results = {}
    embedding_all_results = {}

    for i, query in enumerate(queries, 1):
        query_id = query["id"]
        query_text = query["text"]

        # BM25
        bm25_docs = bm25.retrieve(query_text, top_k=TOP_K_RETRIEVAL)
        bm25_all_results[query_id] = [doc["filename"] for doc in bm25_docs]

        # Embeddings
        embedding_docs = embeddings.retrieve(query_text, top_k=TOP_K_RETRIEVAL)
        embedding_all_results[query_id] = [doc["filename"] for doc in embedding_docs]

        if i % 10 == 0:
            logger.info(f"  Processed {i}/{len(queries)} queries...")

    # Calculate metrics
    logger.info("\nCalculating metrics...")
    bm25_metrics = metrics_calc.calculate_multiple_queries(bm25_all_results, ground_truth)
    embedding_metrics = metrics_calc.calculate_multiple_queries(
        embedding_all_results, ground_truth
    )

    logger.info("\nBM25 Metrics:")
    for metric, value in sorted(bm25_metrics.items()):
        logger.info(f"  {metric:15s}: {value:.4f}")

    logger.info("\nEmbedding Metrics:")
    for metric, value in sorted(embedding_metrics.items()):
        logger.info(f"  {metric:15s}: {value:.4f}")

    # Determine winner
    bm25_avg = sum(bm25_metrics.values()) / len(bm25_metrics)
    embedding_avg = sum(embedding_metrics.values()) / len(embedding_metrics)

    logger.info("\n" + "=" * 80)
    logger.info(f"BM25 Average:       {bm25_avg:.4f}")
    logger.info(f"Embedding Average:  {embedding_avg:.4f}")

    if embedding_avg > bm25_avg + 0.01:
        logger.info(f"✅ EMBEDDING WINS by {(embedding_avg - bm25_avg):.4f} points")
    elif bm25_avg > embedding_avg + 0.01:
        logger.info(f"✅ BM25 WINS by {(bm25_avg - embedding_avg):.4f} points")
    else:
        logger.info("⚖️  TIED")

    logger.info("=" * 80)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_retrievers()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
