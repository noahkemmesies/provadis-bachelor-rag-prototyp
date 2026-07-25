"""Base retriever interface"""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseRetriever(ABC):
    """Abstract base class for retrievers"""

    def __init__(self, documents: List[Dict[str, str]]):
        """
        Initialize retriever with documents

        Args:
            documents: List of document dictionaries with 'content' key
        """
        self.documents = documents
        self.num_documents = len(documents)

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k documents for a query

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        pass

    @abstractmethod
    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Retrieve top-k documents with relevance scores

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        pass

    def get_document_ids_from_results(self, results: List[Dict]) -> List[str]:
        """Extract document IDs/filenames from results"""
        return [doc.get("filename", doc.get("id", "")) for doc in results]
