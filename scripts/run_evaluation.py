#!/usr/bin/env python
"""
Main evaluation script: Compare BM25 vs Embedding-based Retrievers
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    KUBERNETES_DATA_DIR,
    QUERIES_DIR,
    RESULTS_DIR,
    TOP_K_RETRIEVAL,
)
from src.data.loader import load_queries, load_documents
from src.evaluation.metrics import MetricsCalculator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.utils.logger import logger


def run_retriever_evaluation():
    """Run complete evaluation of both retrievers"""

    logger.info("=" * 80)
    logger.info("BM25 vs Embedding-based Retriever Evaluation")
    logger.info("=" * 80)

    # ========================================================================
    # 1. Load Data
    # ========================================================================
    logger.info("\n[1/5] Loading Kubernetes documents...")
    documents = load_documents(KUBERNETES_DATA_DIR)
    logger.info(f"✓ Loaded {len(documents)} YAML documents")

    if not documents:
        logger.error("No documents found! Exiting.")
        return

    # ========================================================================
    # 2. Load Queries & Ground Truth
    # ========================================================================
    logger.info("\n[2/5] Loading queries and ground truth...")
    queries, ground_truth = load_queries(QUERIES_DIR / "queries.json")
    logger.info(f"✓ Loaded {len(queries)} queries")
    logger.info(f"✓ Loaded ground truth for {len(ground_truth)} queries")

    if not queries:
        logger.error("No queries found! Exiting.")
        return

    # ========================================================================
    # 3. Initialize & Test BM25 Retriever
    # ========================================================================
    logger.info("\n[3/5] Initializing BM25 Retriever...")
    try:
        bm25_retriever = BM25Retriever(documents)
        logger.info("✓ BM25 Retriever initialized")

        # Test BM25
        logger.info("Testing BM25 retriever...")
        bm25_results = {}
        for query in queries:
            query_id = query["id"]
            query_text = query["text"]

            retrieved_docs = bm25_retriever.retrieve(query_text, top_k=TOP_K_RETRIEVAL)
            retrieved_ids = [doc["filename"] for doc in retrieved_docs]
            bm25_results[query_id] = retrieved_ids

            if query_id in ["q1", "q2", "q10"]:  # Log sample results
                logger.debug(f"  {query_id}: {retrieved_ids[:3]}")

        logger.info(f"✓ BM25 retrieved results for {len(bm25_results)} queries")

    except Exception as e:
        logger.error(f"Error with BM25 Retriever: {e}")
        bm25_results = {}

    # ========================================================================
    # 4. Initialize & Test Embedding Retriever
    # ========================================================================
    logger.info("\n[4/5] Initializing Embedding Retriever...")
    try:
        embedding_retriever = EmbeddingRetriever(documents)
        logger.info("✓ Embedding Retriever initialized")

        # Test Embeddings
        logger.info("Testing Embedding retriever...")
        embedding_results = {}
        for query in queries:
            query_id = query["id"]
            query_text = query["text"]

            retrieved_docs = embedding_retriever.retrieve(query_text, top_k=TOP_K_RETRIEVAL)
            retrieved_ids = [doc["filename"] for doc in retrieved_docs]
            embedding_results[query_id] = retrieved_ids

            if query_id in ["q1", "q2", "q10"]:  # Log sample results
                logger.debug(f"  {query_id}: {retrieved_ids[:3]}")

        logger.info(f"✓ Embeddings retrieved results for {len(embedding_results)} queries")

    except Exception as e:
        logger.error(f"Error with Embedding Retriever: {e}")
        embedding_results = {}

    # ========================================================================
    # 5. Evaluate Results
    # ========================================================================
    logger.info("\n[5/5] Evaluating results...")

    metrics_calc = MetricsCalculator(k_values=[5, 10])

    # Calculate BM25 metrics
    logger.info("  Calculating BM25 metrics...")
    bm25_metrics = metrics_calc.calculate_multiple_queries(bm25_results, ground_truth)

    # Calculate Embedding metrics
    logger.info("  Calculating Embedding metrics...")
    embedding_metrics = metrics_calc.calculate_multiple_queries(embedding_results, ground_truth)

    # ========================================================================
    # 6. Compare & Report Results
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    # Create comparison dataframe
    comparison_data = {
        "Metric": list(bm25_metrics.keys()),
        "BM25": [bm25_metrics[m] for m in bm25_metrics.keys()],
        "Embedding": [embedding_metrics[m] for m in bm25_metrics.keys()],
    }

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison["Difference"] = df_comparison["Embedding"] - df_comparison["BM25"]
    df_comparison["Winner"] = df_comparison.apply(
        lambda row: "Embedding" if row["Difference"] > 0.01 else ("BM25" if row["Difference"] < -0.01 else "Tie"),
        axis=1,
    )

    logger.info("\nMetrics Comparison:")
    logger.info(df_comparison.to_string(index=False))

    # Summary statistics
    logger.info("\n" + "-" * 80)
    logger.info("SUMMARY")
    logger.info("-" * 80)

    bm25_avg = df_comparison["BM25"].mean()
    embedding_avg = df_comparison["Embedding"].mean()
    embedding_wins = (df_comparison["Winner"] == "Embedding").sum()
    bm25_wins = (df_comparison["Winner"] == "BM25").sum()
    ties = (df_comparison["Winner"] == "Tie").sum()

    logger.info(f"Average BM25 Score:      {bm25_avg:.4f}")
    logger.info(f"Average Embedding Score: {embedding_avg:.4f}")
    logger.info(f"Embedding Wins:          {embedding_wins} metrics")
    logger.info(f"BM25 Wins:               {bm25_wins} metrics")
    logger.info(f"Ties:                    {ties} metrics")

    if embedding_avg > bm25_avg:
        logger.info(f"\n✅ EMBEDDING RETRIEVER is BETTER ({embedding_avg - bm25_avg:.4f} points higher)")
    elif bm25_avg > embedding_avg:
        logger.info(f"\n✅ BM25 RETRIEVER is BETTER ({bm25_avg - embedding_avg:.4f} points higher)")
    else:
        logger.info("\n⚖️  RESULTS ARE TIED")

    # ========================================================================
    # 7. Save Results
    # ========================================================================
    logger.info("\nSaving results...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save comparison
    df_comparison.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)
    logger.info(f"✓ Saved: {RESULTS_DIR / 'metrics_comparison.csv'}")

    # Save detailed results
    results_json = {
        "dataset": {
            "num_documents": len(documents),
            "num_queries": len(queries),
        },
        "bm25": {
            "retrieval_results": bm25_results,
            "metrics": bm25_metrics,
        },
        "embedding": {
            "retrieval_results": embedding_results,
            "metrics": embedding_metrics,
        },
        "comparison": {
            "better": "Embedding" if embedding_avg > bm25_avg else "BM25",
            "margin": float(abs(embedding_avg - bm25_avg)),
            "embedding_wins": int(embedding_wins),
            "bm25_wins": int(bm25_wins),
            "ties": int(ties),
        },
    }

    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"✓ Saved: {RESULTS_DIR / 'evaluation_results.json'}")

    # ========================================================================
    # 8. Sample Queries Analysis
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE QUERY ANALYSIS")
    logger.info("=" * 80)

    sample_queries = queries[:5]  # Show first 5 queries

    for query in sample_queries:
        query_id = query["id"]
        query_text = query["text"]
        relevant_docs = ground_truth.get(query_id, [])

        bm25_retrieved = bm25_results.get(query_id, [])
        embedding_retrieved = embedding_results.get(query_id, [])

        bm25_matches = len(set(bm25_retrieved) & set(relevant_docs))
        embedding_matches = len(set(embedding_retrieved) & set(relevant_docs))

        logger.info(f"\n{query_id}: {query_text[:60]}...")
        logger.info(f"  Relevant docs:      {relevant_docs}")
        logger.info(f"  BM25 matches:       {bm25_matches}/{len(relevant_docs)}")
        logger.info(f"  Embedding matches:  {embedding_matches}/{len(relevant_docs)}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("=" * 80)

    return {
        "comparison": df_comparison,
        "bm25_metrics": bm25_metrics,
        "embedding_metrics": embedding_metrics,
        "results_json": results_json,
    }


if __name__ == "__main__":
    try:
        results = run_retriever_evaluation()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
