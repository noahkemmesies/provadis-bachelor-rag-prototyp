"""Evaluator class for comprehensive evaluation of retrievers"""

import json
from typing import Dict, List, Tuple

import pandas as pd

from src.evaluation.metrics import MetricsCalculator
from src.utils.logger import logger


class RetrieverEvaluator:
    """Evaluate and compare retriever systems"""

    def __init__(self, ground_truth: Dict[str, List[str]], k_values: List[int] = None):
        """
        Initialize evaluator

        Args:
            ground_truth: Dict mapping query_id to list of relevant document IDs
            k_values: List of k values for @k metrics (default: [5, 10])
        """
        self.ground_truth = ground_truth
        self.metrics_calc = MetricsCalculator(k_values=k_values or [5, 10])
        self.results = {}

    def evaluate_retriever(
        self, retriever_name: str, retrieval_results: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate a retriever's results

        Args:
            retriever_name: Name of the retriever (e.g., "BM25", "Embedding")
            retrieval_results: Dict mapping query_id to list of retrieved document IDs

        Returns:
            Dict of metric_name to score
        """
        logger.info(f"Evaluating {retriever_name}...")

        metrics = self.metrics_calc.calculate_multiple_queries(retrieval_results, self.ground_truth)

        self.results[retriever_name] = {
            "retrieval_results": retrieval_results,
            "metrics": metrics,
        }

        return metrics

    def compare_retrievers(self, retriever1: str, retriever2: str) -> pd.DataFrame:
        """
        Compare two retrievers

        Args:
            retriever1: Name of first retriever
            retriever2: Name of second retriever

        Returns:
            DataFrame with comparison metrics
        """
        if retriever1 not in self.results or retriever2 not in self.results:
            raise ValueError(f"Missing results for {retriever1} or {retriever2}")

        metrics1 = self.results[retriever1]["metrics"]
        metrics2 = self.results[retriever2]["metrics"]

        comparison_data = {
            "Metric": list(metrics1.keys()),
            retriever1: [metrics1[m] for m in metrics1.keys()],
            retriever2: [metrics2[m] for m in metrics2.keys()],
        }

        df = pd.DataFrame(comparison_data)
        df["Difference"] = df[retriever2] - df[retriever1]
        df["Winner"] = df.apply(
            lambda row: retriever2
            if row["Difference"] > 0.01
            else (retriever1 if row["Difference"] < -0.01 else "Tie"),
            axis=1,
        )

        return df

    def get_summary(self, retriever1: str, retriever2: str) -> Dict:
        """
        Get summary comparison

        Args:
            retriever1: Name of first retriever
            retriever2: Name of second retriever

        Returns:
            Dict with summary statistics
        """
        metrics1 = self.results[retriever1]["metrics"]
        metrics2 = self.results[retriever2]["metrics"]

        avg1 = sum(metrics1.values()) / len(metrics1)
        avg2 = sum(metrics2.values()) / len(metrics2)

        wins1 = sum(1 for m in metrics1 if metrics1[m] > metrics2[m] + 0.01)
        wins2 = sum(1 for m in metrics2 if metrics2[m] > metrics1[m] + 0.01)
        ties = len(metrics1) - wins1 - wins2

        return {
            retriever1: {
                "average_score": float(avg1),
                "metric_wins": int(wins1),
                "metrics": {k: float(v) for k, v in metrics1.items()},
            },
            retriever2: {
                "average_score": float(avg2),
                "metric_wins": int(wins2),
                "metrics": {k: float(v) for k, v in metrics2.items()},
            },
            "comparison": {
                "better_retriever": retriever2 if avg2 > avg1 else (retriever1 if avg1 > avg2 else "Tie"),
                "margin": float(abs(avg2 - avg1)),
                f"{retriever1}_wins": int(wins1),
                f"{retriever2}_wins": int(wins2),
                "ties": int(ties),
            },
        }

    def get_per_query_analysis(
        self, retriever1: str, retriever2: str, query_ids: List[str] = None
    ) -> pd.DataFrame:
        """
        Get per-query analysis

        Args:
            retriever1: Name of first retriever
            retriever2: Name of second retriever
            query_ids: Specific queries to analyze (default: all)

        Returns:
            DataFrame with per-query metrics
        """
        if retriever1 not in self.results or retriever2 not in self.results:
            raise ValueError(f"Missing results for {retriever1} or {retriever2}")

        results1 = self.results[retriever1]["retrieval_results"]
        results2 = self.results[retriever2]["retrieval_results"]

        if query_ids is None:
            query_ids = list(self.ground_truth.keys())

        analysis_data = []

        for query_id in query_ids:
            if query_id not in self.ground_truth:
                continue

            relevant_ids = set(self.ground_truth[query_id])
            retrieved1 = set(results1.get(query_id, []))
            retrieved2 = set(results2.get(query_id, []))

            matches1 = len(retrieved1 & relevant_ids)
            matches2 = len(retrieved2 & relevant_ids)
            both_found = len(retrieved1 & retrieved2 & relevant_ids)

            analysis_data.append({
                "Query ID": query_id,
                f"{retriever1} Matches": matches1,
                f"{retriever2} Matches": matches2,
                "Both Found": both_found,
                "Better": retriever2 if matches2 > matches1 else (retriever1 if matches1 > matches2 else "Tie"),
            })

        return pd.DataFrame(analysis_data)

    def export_results(self, output_dir: str) -> None:
        """
        Export all results to JSON

        Args:
            output_dir: Directory to save results
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        export_data = {}
        for retriever_name, data in self.results.items():
            export_data[retriever_name] = {
                "metrics": {k: float(v) for k, v in data["metrics"].items()},
                "retrieval_results": data["retrieval_results"],
            }

        with open(output_path / "detailed_results.json", "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported results to {output_path / 'detailed_results.json'}")
