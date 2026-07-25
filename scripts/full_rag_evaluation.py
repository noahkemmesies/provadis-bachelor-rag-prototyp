#!/usr/bin/env python
"""Full RAG Pipeline Evaluation on all queries"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import KUBERNETES_DATA_DIR, QUERIES_DIR, RESULTS_DIR
from src.data.loader import load_queries, load_documents
from src.evaluation.answer_quality import AnswerQualityEvaluator
from src.evaluation.metrics import MetricsCalculator
from src.rag.llm_integration import OllamaLLM
from src.rag.pipeline import HybridRAGPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.utils.logger import logger


def run_full_evaluation():
    """Run full RAG evaluation on all 40 queries"""

    logger.info("=" * 80)
    logger.info("FULL RAG PIPELINE EVALUATION")
    logger.info("=" * 80)

    # ========================================================================
    # 1. Check Prerequisites
    # ========================================================================
    logger.info("\n[1/7] Checking prerequisites...")

    llm = OllamaLLM()
    if not llm.check_connection():
        logger.error("Cannot connect to Ollama! Exiting.")
        return False

    # ========================================================================
    # 2. Load Data
    # ========================================================================
    logger.info("\n[2/7] Loading data...")
    documents = load_documents(KUBERNETES_DATA_DIR)
    queries, ground_truth = load_queries(QUERIES_DIR / "queries.json")

    logger.info(f"✓ Loaded {len(documents)} documents")
    logger.info(f"✓ Loaded {len(queries)} queries")

    if not documents or not queries:
        logger.error("Missing data!")
        return False

    # ========================================================================
    # 3. Initialize Systems
    # ========================================================================
    logger.info("\n[3/7] Initializing retrieval systems...")

    try:
        bm25_retriever = BM25Retriever(documents)
        logger.info("✓ BM25 Retriever initialized")
    except Exception as e:
        logger.error(f"BM25 failed: {e}")
        return False

    try:
        logger.info("Initializing Embedding Retriever (may take a moment)...")
        embedding_retriever = EmbeddingRetriever(documents)
        logger.info("✓ Embedding Retriever initialized")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        logger.warning("Continuing with BM25 only")
        embedding_retriever = None

    # ========================================================================
    # 4. Run RAG Pipelines on All Queries
    # ========================================================================
    logger.info("\n[4/7] Running RAG pipelines on all 40 queries...")
    logger.info("This may take 10-30 minutes...")

    retrievers = {"BM25": bm25_retriever}
    if embedding_retriever:
        retrievers["Embedding"] = embedding_retriever

    hybrid_pipeline = HybridRAGPipeline(retrievers, llm)

    rag_results = {}
    for i, query in enumerate(queries, 1):
        query_id = query["id"]
        query_text = query["text"]

        try:
            logger.debug(f"  [{i}/{len(queries)}] Processing {query_id}...")
            results = hybrid_pipeline.run(query_text)
            rag_results[query_id] = results

            if i % 5 == 0:
                logger.info(f"  Progress: {i}/{len(queries)} queries processed")

        except Exception as e:
            logger.error(f"Error processing {query_id}: {e}")
            rag_results[query_id] = {
                "error": str(e),
                "query": query_text,
            }

    logger.info(f"✓ Completed RAG on {len(rag_results)} queries")

    # ========================================================================
    # 5. Evaluate Answer Quality
    # ========================================================================
    logger.info("\n[5/7] Evaluating answer quality...")

    quality_evaluator = AnswerQualityEvaluator()
    quality_scores = {}

    for query_id, results in rag_results.items():
        query_text = queries[next(i for i, q in enumerate(queries) if q["id"] == query_id)]["text"]

        for retriever_name, rag_result in results.items():
            if "error" in rag_result:
                continue

            answer = rag_result.get("answer", "")
            docs = rag_result.get("retrieved_documents", [])

            try:
                quality = quality_evaluator.evaluate_answer(answer, query_text, docs)

                if query_id not in quality_scores:
                    quality_scores[query_id] = {}

                quality_scores[query_id][retriever_name] = quality["overall_score"]

            except Exception as e:
                logger.error(f"Quality evaluation error for {query_id}/{retriever_name}: {e}")

    logger.info(f"✓ Evaluated {len(quality_scores)} queries")

    # ========================================================================
    # 6. Calculate Retrieval Metrics
    # ========================================================================
    logger.info("\n[6/7] Calculating retrieval metrics...")

    metrics_calc = MetricsCalculator(k_values=[5, 10])

    retrieval_results = {}
    for retriever_name in retrievers.keys():
        results_for_retriever = {}
        for query_id, rag_result in rag_results.items():
            if query_id not in rag_result or "error" in rag_result:
                continue

            retriever_result = rag_result.get(retriever_name, {})
            if "retrieved_documents" in retriever_result:
                docs = retriever_result["retrieved_documents"]
                results_for_retriever[query_id] = [d["filename"] for d in docs]

        metrics = metrics_calc.calculate_multiple_queries(results_for_retriever, ground_truth)
        retrieval_results[retriever_name] = {
            "metrics": metrics,
            "retrieval_results": results_for_retriever,
        }

    logger.info(f"✓ Calculated metrics for {len(retrieval_results)} retrievers")

    # ========================================================================
    # 7. Generate Report
    # ========================================================================
    logger.info("\n[7/7] Generating comprehensive report...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save RAG Results
    with open(RESULTS_DIR / "rag_results_all_queries.json", "w") as f:
        # Convert to serializable format
        serializable_results = {}
        for qid, results in rag_results.items():
            if "error" not in results:
                serializable_results[qid] = {
                    retriever: {
                        "answer": r.get("answer", "")[:500],  # Truncate for storage
                        "num_documents": len(r.get("retrieved_documents", [])),
                    }
                    for retriever, r in results.items()
                }
        json.dump(serializable_results, f, indent=2)

    logger.info(f"✓ Saved RAG results to {RESULTS_DIR / 'rag_results_all_queries.json'}")

    # Save Quality Scores
    with open(RESULTS_DIR / "quality_scores.json", "w") as f:
        json.dump(quality_scores, f, indent=2)

    logger.info(f"✓ Saved quality scores to {RESULTS_DIR / 'quality_scores.json'}")

    # Save Retrieval Metrics
    metrics_report = {}
    for retriever_name, data in retrieval_results.items():
        metrics_report[retriever_name] = {
            "metrics": {k: float(v) for k, v in data["metrics"].items()}
        }

    with open(RESULTS_DIR / "retrieval_metrics_full.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    logger.info(f"✓ Saved retrieval metrics to {RESULTS_DIR / 'retrieval_metrics_full.json'}")

    # ========================================================================
    # 8. Summary Report
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)

    # Quality scores summary
    logger.info("\nAnswer Quality Scores (0-1 scale):")
    if quality_scores:
        all_scores = {}
        for scores in quality_scores.values():
            for retriever, score in scores.items():
                if retriever not in all_scores:
                    all_scores[retriever] = []
                all_scores[retriever].append(score)

        for retriever, scores in all_scores.items():
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            logger.info(
                f"  {retriever}:")
            logger.info(f"    Average: {avg_score:.4f}")
            logger.info(f"    Min:     {min_score:.4f}")
            logger.info(f"    Max:     {max_score:.4f}")

    # Retrieval metrics summary
    logger.info("\nRetrieval Metrics:")
    for retriever_name, data in retrieval_results.items():
        metrics = data["metrics"]
        logger.info(f"  {retriever_name}:")
        for metric_name in sorted(metrics.keys()):
            logger.info(f"    {metric_name:15s}: {metrics[metric_name]:.4f}")

    # Comparison
    logger.info("\n" + "-" * 80)
    logger.info("COMPARISON")
    logger.info("-" * 80)

    retriever_names = list(retrieval_results.keys())
    if len(retriever_names) >= 2:
        r1, r2 = retriever_names[0], retriever_names[1]
        metrics1 = retrieval_results[r1]["metrics"]
        metrics2 = retrieval_results[r2]["metrics"]

        r1_wins = sum(1 for m in metrics1 if metrics1[m] > metrics2[m] + 0.01)
        r2_wins = sum(1 for m in metrics2 if metrics2[m] > metrics1[m] + 0.01)

        logger.info(f"\n{r1} vs {r2}:")
        logger.info(f"  {r1} metric wins: {r1_wins}")
        logger.info(f"  {r2} metric wins: {r2_wins}")

        avg1 = sum(metrics1.values()) / len(metrics1)
        avg2 = sum(metrics2.values()) / len(metrics2)

        winner = r1 if avg1 > avg2 else r2
        margin = abs(avg1 - avg2)
        logger.info(f"\n  Winner: {winner} (margin: {margin:.4f})")

    logger.info("\n" + "=" * 80)
    logger.info("✅ FULL EVALUATION COMPLETE")
    logger.info("=" * 80)

    logger.info("\nOutput files:")
    logger.info(f"  - {RESULTS_DIR / 'rag_results_all_queries.json'}")
    logger.info(f"  - {RESULTS_DIR / 'quality_scores.json'}")
    logger.info(f"  - {RESULTS_DIR / 'retrieval_metrics_full.json'}")

    logger.info("\nNext steps:")
    logger.info("  - Analyze results in Jupyter notebooks")
    logger.info("  - Generate visualizations")
    logger.info("  - Write comprehensive report")

    return True


if __name__ == "__main__":
    try:
        success = run_full_evaluation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
