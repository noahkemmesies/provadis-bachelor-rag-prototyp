"""BM25 Retriever implementation"""

from typing import Dict, List

from rank_bm25 import BM25Okapi

from src.config import BM25_B, BM25_K1
from src.retrieval.base import BaseRetriever
from src.utils.logger import logger


class BM25Retriever(BaseRetriever):
    """BM25-based retriever using rank_bm25 library"""

    def __init__(self, documents: List[Dict[str, str]], k1: float = BM25_K1, b: float = BM25_B):
        """
        Initialize BM25 retriever

        Args:
            documents: List of document dictionaries with 'content' key
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        super().__init__(documents)

        # Tokenize documents
        self.tokenized_docs = [self._tokenize(doc["content"]) for doc in documents]

        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=k1, b=b)

        logger.info(f"Initialized BM25Retriever with {self.num_documents} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace"""
        return text.lower().split()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k documents for a query

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["score"] = scores[idx]
            doc["rank"] = len(results) + 1
            results.append(doc)

        return results

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Retrieve top-k documents with relevance scores

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        results = self.retrieve(query, top_k)
        return [(doc, doc["score"]) for doc in results]

    def get_all_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for all documents

        Args:
            query: Query string

        Returns:
            List of scores for all documents
        """
        query_tokens = self._tokenize(query)
        return self.bm25.get_scores(query_tokens)
