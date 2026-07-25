"""Tests for evaluation metrics"""

import pytest

from src.evaluation.metrics import (
    MetricsCalculator,
    average_precision,
    f1_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionRecall:
    """Test Precision and Recall metrics"""

    def test_precision_at_5(self):
        """Test Precision@5"""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc5"]
        assert precision_at_k(retrieved, relevant, 5) == pytest.approx(0.6)

    def test_precision_at_3(self):
        """Test Precision@3"""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc5"]
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2.0 / 3.0)

    def test_recall_at_5(self):
        """Test Recall@5"""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc5"]
        assert recall_at_k(retrieved, relevant, 5) == pytest.approx(1.0)

    def test_recall_at_3(self):
        """Test Recall@3"""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc5"]
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2.0 / 3.0)

    def test_precision_no_relevant(self):
        """Test with no relevant documents"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = []
        assert precision_at_k(retrieved, relevant, 5) == pytest.approx(0.0)

    def test_recall_no_retrieved(self):
        """Test with no relevant documents found"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4", "doc5"]
        assert recall_at_k(retrieved, relevant, 5) == pytest.approx(0.0)


class TestMRR:
    """Test Mean Reciprocal Rank"""

    def test_mrr_first_position(self):
        """Test MRR when relevant doc is first"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1"]
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)

    def test_mrr_second_position(self):
        """Test MRR when relevant doc is second"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2"]
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_mrr_no_relevant(self):
        """Test MRR when no relevant doc found"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4"]
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.0)


class TestNDCG:
    """Test Normalized Discounted Cumulative Gain"""

    def test_ndcg_perfect(self):
        """Test nDCG with perfect ranking"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2"]
        ndcg = ndcg_at_k(retrieved, relevant, 3)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_imperfect(self):
        """Test nDCG with imperfect ranking"""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = ["doc1", "doc3"]
        ndcg = ndcg_at_k(retrieved, relevant, 4)
        # Should be less than 1.0 due to suboptimal ranking
        assert 0.0 < ndcg < 1.0


class TestF1:
    """Test F1@k metric"""

    def test_f1_perfect(self):
        """Test F1 with perfect precision and recall"""
        retrieved = ["doc1", "doc2"]
        relevant = ["doc1", "doc2"]
        f1 = f1_at_k(retrieved, relevant, 2)
        assert f1 == pytest.approx(1.0)

    def test_f1_no_match(self):
        """Test F1 with no matches"""
        retrieved = ["doc1", "doc2"]
        relevant = ["doc3", "doc4"]
        f1 = f1_at_k(retrieved, relevant, 2)
        assert f1 == pytest.approx(0.0)


class TestAveragePrecision:
    """Test Average Precision"""

    def test_ap_perfect(self):
        """Test AP with perfect ranking"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        ap = average_precision(retrieved, relevant)
        assert ap == pytest.approx(1.0)

    def test_ap_partial(self):
        """Test AP with partial matches"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc3"]
        # AP = (1/1 + 2/3) / 2
        ap = average_precision(retrieved, relevant)
        assert ap == pytest.approx((1.0 + 2.0/3.0) / 2.0)


class TestMetricsCalculator:
    """Test MetricsCalculator"""

    def test_calculate_all(self):
        """Test calculating all metrics"""
        calculator = MetricsCalculator(k_values=[5, 10])
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3"]

        metrics = calculator.calculate_all(retrieved, relevant)

        assert "precision@5" in metrics
        assert "recall@5" in metrics
        assert "f1@5" in metrics
        assert "ndcg@5" in metrics
        assert "mrr" in metrics
        assert "ap" in metrics

    def test_calculate_multiple_queries(self):
        """Test calculating metrics for multiple queries"""
        calculator = MetricsCalculator(k_values=[5])

        query_results = {
            "q1": ["doc1", "doc2", "doc3"],
            "q2": ["doc4", "doc5", "doc1"],
        }

        ground_truth = {
            "q1": ["doc1", "doc2"],
            "q2": ["doc1", "doc4"],
        }

        avg_metrics = calculator.calculate_multiple_queries(query_results, ground_truth)

        assert "precision@5" in avg_metrics
        assert isinstance(avg_metrics["precision@5"], float)
