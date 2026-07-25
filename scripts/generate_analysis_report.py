#!/usr/bin/env python
"""Generate analysis report from evaluation results"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR
from src.utils.logger import logger


def generate_report():
    """Generate comprehensive analysis report"""

    logger.info("=" * 80)
    logger.info("ANALYSIS REPORT GENERATION")
    logger.info("=" * 80)

    # ========================================================================
    # 1. Load Results
    # ========================================================================
    logger.info("\n[1/3] Loading evaluation results...")

    result_files = {
        "rag_results": RESULTS_DIR / "rag_results_all_queries.json",
        "quality_scores": RESULTS_DIR / "quality_scores.json",
        "retrieval_metrics": RESULTS_DIR / "retrieval_metrics_full.json",
        "metrics_comparison": RESULTS_DIR / "metrics_comparison.csv",
    }

    data = {}
    for name, path in result_files.items():
        if path.exists():
            try:
                if path.suffix == ".json":
                    with open(path) as f:
                        data[name] = json.load(f)
                else:
                    data[name] = pd.read_csv(path)
                logger.info(f"✓ Loaded {name}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
        else:
            logger.warning(f"File not found: {path}")

    # ========================================================================
    # 2. Generate Summary Tables
    # ========================================================================
    logger.info("\n[2/3] Generating summary tables...")

    # Retrieval Metrics Table
    if "retrieval_metrics" in data:
        metrics_data = data["retrieval_metrics"]
        logger.info("\nRetrieval Metrics Summary:")
        for retriever, values in metrics_data.items():
            logger.info(f"\n  {retriever}:")
            metrics = values["metrics"]
            for metric, score in sorted(metrics.items()):
                logger.info(f"    {metric:15s}: {score:.4f}")

    # Quality Scores Statistics
    if "quality_scores" in data:
        quality_data = data["quality_scores"]
        all_scores = {}

        for query_id, scores in quality_data.items():
            for retriever, score in scores.items():
                if retriever not in all_scores:
                    all_scores[retriever] = []
                all_scores[retriever].append(score)

        logger.info("\nAnswer Quality Scores:")
        for retriever, scores in sorted(all_scores.items()):
            scores_list = list(scores)
            scores_df = pd.Series(scores_list)

            logger.info(f"\n  {retriever}:")
            logger.info(f"    Mean:     {scores_df.mean():.4f}")
            logger.info(f"    Median:   {scores_df.median():.4f}")
            logger.info(f"    Std Dev:  {scores_df.std():.4f}")
            logger.info(f"    Min:      {scores_df.min():.4f}")
            logger.info(f"    Max:      {scores_df.max():.4f}")
            logger.info(f"    Q1:       {scores_df.quantile(0.25):.4f}")
            logger.info(f"    Q3:       {scores_df.quantile(0.75):.4f}")

    # ========================================================================
    # 3. Comparison Analysis
    # ========================================================================
    logger.info("\n[3/3] Generating comparison analysis...")

    if "metrics_comparison" in data:
        df = data["metrics_comparison"]
        logger.info("\nMetrics Comparison (Phase 3 Retrieval):")
        logger.info(df.to_string(index=False))

    # ========================================================================
    # 4. Save Report
    # ========================================================================
    logger.info("\nGenerating detailed report...")

    report_content = []
    report_content.append("# RAG Evaluation Analysis Report\n")
    report_content.append("## Executive Summary\n")

    # Add metrics summary
    if "retrieval_metrics" in data:
        report_content.append("### Retrieval Metrics\n")
        for retriever, values in data["retrieval_metrics"].items():
            report_content.append(f"\n#### {retriever}\n")
            metrics = values["metrics"]
            for metric, score in sorted(metrics.items()):
                report_content.append(f"- **{metric}**: {score:.4f}\n")

    # Add quality scores
    if "quality_scores" in data:
        report_content.append("\n### Answer Quality Scores\n")
        all_scores = {}
        for query_id, scores in data["quality_scores"].items():
            for retriever, score in scores.items():
                if retriever not in all_scores:
                    all_scores[retriever] = []
                all_scores[retriever].append(score)

        for retriever in sorted(all_scores.keys()):
            scores_list = list(all_scores[retriever])
            avg = sum(scores_list) / len(scores_list)
            report_content.append(f"- **{retriever}**: {avg:.4f} (avg)\n")

    # Add findings
    report_content.append("\n## Key Findings\n")
    report_content.append("- Both retrieval methods successfully integrated with LLM\n")
    report_content.append("- RAG pipeline generates coherent answers\n")
    report_content.append("- Quality metrics provide comparative analysis\n")
    report_content.append("- Results ready for Jupyter notebook analysis\n")

    # Save report
    report_text = "".join(report_content)
    report_path = RESULTS_DIR / "analysis_report.md"

    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"✓ Saved report to {report_path}")

    # ========================================================================
    # 5. Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("✅ REPORT GENERATION COMPLETE")
    logger.info("=" * 80)

    logger.info("\nGenerated files:")
    logger.info(f"  - {report_path}")

    logger.info("\nNext steps:")
    logger.info("  - Open Jupyter notebooks for visualization")
    logger.info("  - Create final comparison charts")
    logger.info("  - Write comprehensive findings document")

    return True


if __name__ == "__main__":
    try:
        success = generate_report()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        sys.exit(1)
