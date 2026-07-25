"""Evaluation metrics for information retrieval"""

from typing import Dict, List

import numpy as np

from src.utils.logger import logger


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Precision@k

    Precision@k = (# of relevant documents in top-k) / k

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Top-k value

    Returns:
        Precision@k score
    """
    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_ids))
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Recall@k

    Recall@k = (# of relevant documents in top-k) / (total # of relevant documents)

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Top-k value

    Returns:
        Recall@k score
    """
    if len(relevant_ids) == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_ids))
    return relevant_in_top_k / len(relevant_ids)


def f1_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate F1@k

    F1@k = 2 * (Precision@k * Recall@k) / (Precision@k + Recall@k)

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Top-k value

    Returns:
        F1@k score
    """
    precision = precision_at_k(retrieved_ids, relevant_ids, k)
    recall = recall_at_k(retrieved_ids, relevant_ids, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)

    MRR = 1 / (position of first relevant document)

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs

    Returns:
        MRR score
    """
    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@k)

    nDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Top-k value

    Returns:
        nDCG@k score
    """
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # Calculate DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # log2(i+2) because position starts from 1

    # Calculate IDCG@k (ideal DCG)
    num_relevant = len(relevant_ids)
    idcg = 0.0
    for i in range(min(k, num_relevant)):
        idcg += 1.0 / np.log2(i + 2)

    # Calculate nDCG
    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate Average Precision (AP)

    AP = sum(P@k * rel@k) / number of relevant documents

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs

    Returns:
        AP score
    """
    if len(relevant_ids) == 0:
        return 0.0

    relevant_set = set(relevant_ids)
    score = 0.0
    num_hits = 0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            score += precision_at_i

    return score / len(relevant_ids)


class MetricsCalculator:
    """Calculate multiple evaluation metrics at once"""

    def __init__(self, k_values: List[int] = None):
        """
        Initialize metrics calculator

        Args:
            k_values: List of k values for @k metrics (default: [5, 10])
        """
        self.k_values = k_values or [5, 10]

    def calculate_all(self, retrieved_ids: List[str],
                      relevant_ids: List[str]) -> Dict[str, float]:
        """
        Calculate all metrics

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs

        Returns:
            Dictionary of metric names to scores
        """
        metrics = {}

        # @k metrics
        for k in self.k_values:
            metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"f1@{k}"] = f1_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

        # Non-k metrics
        metrics["mrr"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)
        metrics["ap"] = average_precision(retrieved_ids, relevant_ids)

        return metrics

    def calculate_multiple_queries(self, query_results: Dict[str, List[str]],
                                   ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate metrics averaged over multiple queries

        Args:
            query_results: Dict mapping query_id to retrieved document IDs
            ground_truth: Dict mapping query_id to relevant document IDs

        Returns:
            Dictionary of average metric scores
        """
        all_metrics = {}

        for query_id, retrieved_ids in query_results.items():
            relevant_ids = ground_truth.get(query_id, [])
            metrics = self.calculate_all(retrieved_ids, relevant_ids)

            for metric_name, score in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(score)

        # Average the metrics
        average_metrics = {
            name: np.mean(scores) for name, scores in all_metrics.items()
        }

        return average_metrics
