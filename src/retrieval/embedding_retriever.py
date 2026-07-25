"""Embedding-based Retriever implementation using ChromaDB"""

from typing import Dict, List

from sentence_transformers import SentenceTransformer

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_DEVICE, EMBEDDING_MODEL
from src.retrieval.base import BaseRetriever
from src.utils.logger import logger

try:
    import chromadb
except ImportError:
    logger.warning("chromadb not installed. Install with: pip install chromadb")
    chromadb = None


class EmbeddingRetriever(BaseRetriever):
    """Embedding-based retriever using Sentence Transformers and ChromaDB"""

    def __init__(self, documents: List[Dict[str, str]], model_name: str = EMBEDDING_MODEL,
                 device: str = EMBEDDING_DEVICE, use_persistence: bool = True):
        """
        Initialize Embedding retriever

        Args:
            documents: List of document dictionaries with 'content' key
            model_name: Sentence Transformers model name
            device: Device to use ('cpu' or 'cuda')
            use_persistence: Whether to persist the collection
        """
        super().__init__(documents)

        if chromadb is None:
            raise ImportError("chromadb is not installed")

        # Load embedding model
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

        # Initialize ChromaDB
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        if use_persistence:
            self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        else:
            self.client = chromadb.EphemeralClient()

        # Create or get collection
        self.collection_name = "kubernetes_docs"
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection '{self.collection_name}'")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection '{self.collection_name}'")

        # Embed and add documents
        self._embed_and_add_documents(documents)

        logger.info(f"Initialized EmbeddingRetriever with {self.num_documents} documents "
                    f"using model '{model_name}'")

    def _embed_and_add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Embed documents and add them to ChromaDB"""
        document_texts = [doc["content"] for doc in documents]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(document_texts, show_progress_bar=True)

        # Add to collection
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [{"filename": doc["filename"], "filepath": doc.get("filepath", "")}
                     for doc in documents]

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=document_texts
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k documents for a query

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        # Embed query
        query_embedding = self.model.encode(query)

        # Query collection
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

        # Process results
        retrieved_docs = []
        if results["documents"] and len(results["documents"]) > 0:
            for i, (doc_text, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score (1 / (1 + distance))
                score = 1.0 / (1.0 + distance)

                doc = {
                    "filename": metadata.get("filename", "unknown"),
                    "filepath": metadata.get("filepath", ""),
                    "content": doc_text,
                    "score": score,
                    "rank": i + 1,
                }
                retrieved_docs.append(doc)

        return retrieved_docs

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

    def clear_collection(self) -> None:
        """Clear the ChromaDB collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
