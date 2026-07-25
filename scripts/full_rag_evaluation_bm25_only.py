#!/usr/bin/env python
"""Full RAG Pipeline Evaluation - BM25 Only (without Embeddings)"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import KUBERNETES_DATA_DIR, QUERIES_DIR, RESULTS_DIR
from src.data.loader import load_queries, load_documents
from src.evaluation.answer_quality import AnswerQualityEvaluator
from src.evaluation.metrics import MetricsCalculator
from src.rag.llm_integration import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.logger import logger


def run_bm25_evaluation():
    """Run RAG evaluation with BM25 only"""

    logger.info("=" * 80)
    logger.info("FULL RAG PIPELINE EVALUATION - BM25 ONLY")
    logger.info("=" * 80)

    # ========================================================================
    # 1. Check Ollama
    # ========================================================================
    logger.info("\n[1/6] Checking Ollama connection...")
    llm = OllamaLLM()
    if not llm.check_connection():
        logger.error("Ollama not running! Start it with: ollama serve")
        return False

    # ========================================================================
    # 2. Load Data
    # ========================================================================
    logger.info("\n[2/6] Loading data...")
    documents = load_documents(KUBERNETES_DATA_DIR)
    queries, ground_truth = load_queries(QUERIES_DIR / "queries.json")
    logger.info(f"✓ Loaded {len(documents)} documents & {len(queries)} queries")

    # ========================================================================
    # 3. Initialize BM25
    # ========================================================================
    logger.info("\n[3/6] Initializing BM25 Retriever...")
    bm25_retriever = BM25Retriever(documents)
    logger.info("✓ BM25 ready")

    # ========================================================================
    # 4. Run RAG Pipeline
    # ========================================================================
    logger.info("\n[4/6] Running RAG on all 40 queries...")
    logger.info("(This takes 10-30 minutes - generating answers with Mistral)")

    rag_pipeline = RAGPipeline(bm25_retriever, llm)
    rag_results = {}
    quality_scores = {}

    for i, query in enumerate(queries, 1):
        query_id = query["id"]
        query_text = query["text"]

        try:
            # Generate answer
            result = rag_pipeline.run(query_text)
            rag_results[query_id] = result

            # Evaluate quality
            answer = result.get("answer", "")
            docs = result.get("retrieved_documents", [])
            quality = AnswerQualityEvaluator().evaluate_answer(answer, query_text, docs)
            quality_scores[query_id] = quality["overall_score"]

            if i % 5 == 0:
                logger.info(f"  Progress: {i}/{len(queries)} queries processed")

        except Exception as e:
            logger.error(f"Error on {query_id}: {e}")

    logger.info(f"✓ Completed {len(rag_results)} queries")

    # ========================================================================
    # 5. Calculate Metrics
    # ========================================================================
    logger.info("\n[5/6] Calculating metrics...")

    metrics_calc = MetricsCalculator(k_values=[5, 10])
    bm25_retrieval_results = {}

    for query_id, rag_result in rag_results.items():
        if "error" not in rag_result:
            docs = rag_result.get("retrieved_documents", [])
            bm25_retrieval_results[query_id] = [d["filename"] for d in docs]

    metrics = metrics_calc.calculate_multiple_queries(bm25_retrieval_results, ground_truth)
    logger.info("✓ Metrics calculated")

    # ========================================================================
    # 6. Save & Report
    # ========================================================================
    logger.info("\n[6/6] Saving results...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(RESULTS_DIR / "bm25_rag_results.json", "w") as f:
        json.dump(
            {
                qid: {
                    "answer": r.get("answer", "")[:500],
                    "num_documents": len(r.get("retrieved_documents", [])),
                }
                for qid, r in rag_results.items()
            },
            f,
            indent=2,
        )

    with open(RESULTS_DIR / "bm25_quality_scores.json", "w") as f:
        json.dump(quality_scores, f, indent=2)

    with open(RESULTS_DIR / "bm25_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # ========================================================================
    # 7. Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("=" * 80)

    avg_quality = sum(quality_scores.values()) / len(quality_scores)
    logger.info(f"\nResults:")
    logger.info(f"  Queries processed: {len(rag_results)}")
    logger.info(f"  Average quality score: {avg_quality:.4f}")
    logger.info(f"\nMetrics:")
    for metric, score in sorted(metrics.items()):
        logger.info(f"  {metric:15s}: {score:.4f}")

    logger.info(f"\nFiles saved:")
    logger.info(f"  - {RESULTS_DIR / 'bm25_rag_results.json'}")
    logger.info(f"  - {RESULTS_DIR / 'bm25_quality_scores.json'}")
    logger.info(f"  - {RESULTS_DIR / 'bm25_metrics.json'}")

    return True


if __name__ == "__main__":
    try:
        success = run_bm25_evaluation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
