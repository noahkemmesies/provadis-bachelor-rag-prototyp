"""Tests for retrieval systems"""

import pytest

from src.retrieval.bm25_retriever import BM25Retriever


class TestBM25Retriever:
    """Test BM25Retriever"""

    def test_bm25_initialization(self, sample_documents):
        """Test BM25 initialization"""
        retriever = BM25Retriever(sample_documents)
        assert retriever.num_documents == 3

    def test_bm25_retrieve(self, sample_documents):
        """Test BM25 retrieval"""
        retriever = BM25Retriever(sample_documents)
        results = retriever.retrieve("Deployment", top_k=2)

        assert len(results) <= 2
        assert all("filename" in doc for doc in results)
        assert all("score" in doc for doc in results)

    def test_bm25_scores_descending(self, sample_documents):
        """Test that BM25 scores are in descending order"""
        retriever = BM25Retriever(sample_documents)
        results = retriever.retrieve("nginx", top_k=3)

        scores = [doc["score"] for doc in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_empty_query(self, sample_documents):
        """Test BM25 with empty query"""
        retriever = BM25Retriever(sample_documents)
        results = retriever.retrieve("", top_k=5)
        # Should still return results, just with low scores
        assert isinstance(results, list)

    def test_bm25_top_k(self, sample_documents):
        """Test BM25 respects top_k parameter"""
        retriever = BM25Retriever(sample_documents)

        results_1 = retriever.retrieve("Service", top_k=1)
        results_2 = retriever.retrieve("Service", top_k=2)
        results_3 = retriever.retrieve("Service", top_k=10)

        assert len(results_1) == 1
        assert len(results_2) == 2
        assert len(results_3) == 3  # Only 3 documents total

    def test_bm25_retrieve_with_scores(self, sample_documents):
        """Test BM25 retrieve_with_scores"""
        retriever = BM25Retriever(sample_documents)
        results = retriever.retrieve_with_scores("Pod", top_k=2)

        assert len(results) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(isinstance(score, float) for _, score in results)


class TestEmbeddingRetriever:
    """Test EmbeddingRetriever"""

    @pytest.mark.slow
    def test_embedding_initialization(self, sample_documents):
        """Test Embedding initialization"""
        pytest.importorskip("chromadb")
        from src.retrieval.embedding_retriever import EmbeddingRetriever

        retriever = EmbeddingRetriever(sample_documents)
        assert retriever.num_documents == 3

    @pytest.mark.slow
    def test_embedding_retrieve(self, sample_documents):
        """Test Embedding retrieval"""
        pytest.importorskip("chromadb")
        from src.retrieval.embedding_retriever import EmbeddingRetriever

        retriever = EmbeddingRetriever(sample_documents)
        results = retriever.retrieve("Deployment", top_k=2)

        assert len(results) <= 2
        assert all("filename" in doc for doc in results)
        assert all("score" in doc for doc in results)

    @pytest.mark.slow
    def test_embedding_scores_descending(self, sample_documents):
        """Test that Embedding scores are in descending order"""
        pytest.importorskip("chromadb")
        from src.retrieval.embedding_retriever import EmbeddingRetriever

        retriever = EmbeddingRetriever(sample_documents)
        results = retriever.retrieve("nginx", top_k=3)

        scores = [doc["score"] for doc in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.slow
    def test_embedding_top_k(self, sample_documents):
        """Test Embedding respects top_k parameter"""
        pytest.importorskip("chromadb")
        from src.retrieval.embedding_retriever import EmbeddingRetriever

        retriever = EmbeddingRetriever(sample_documents)

        results_1 = retriever.retrieve("Service", top_k=1)
        results_2 = retriever.retrieve("Service", top_k=2)
        results_3 = retriever.retrieve("Service", top_k=10)

        assert len(results_1) == 1
        assert len(results_2) == 2
        assert len(results_3) == 3


class TestRetrieverComparison:
    """Test comparing retrievers"""

    @pytest.mark.slow
    def test_both_retrievers_return_results(self, sample_documents):
        """Test that both retrievers return results for same query"""
        pytest.importorskip("chromadb")
        from src.retrieval.embedding_retriever import EmbeddingRetriever

        bm25 = BM25Retriever(sample_documents)
        embedding = EmbeddingRetriever(sample_documents)

        query = "What is Kubernetes?"
        bm25_results = bm25.retrieve(query, top_k=2)
        embedding_results = embedding.retrieve(query, top_k=2)

        assert len(bm25_results) > 0
        assert len(embedding_results) > 0
